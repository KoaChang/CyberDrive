import json
import os
import asyncio
import aiohttp
import aiofiles
from typing import Optional
from dotenv import load_dotenv

async def process_single_file(file_path: str, session: aiohttp.ClientSession) -> Optional[tuple[str, str]]:
    """
    Process a single JSON file and extract the answer choice using OpenAI API.
    
    Args:
        file_path: Path to the JSON file
        session: aiohttp ClientSession for making API calls
        
    Returns:
        Tuple of (filename, answer) if successful, None if failed
    """
    try:
        # Read the JSON file asynchronously
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            data = json.loads(content)
            
        # Extract the answer text
        answer_text = data.get('answer', '')
        if not answer_text:
            print(f"Warning: No answer found in {file_path}")
            return None
            
        # Create the API call to GPT-4
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY_KOA_4o')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "Extract the letter choice (A, B, C, D, or E) that is indicated as the answer in the text. Output the answer in tags like this: <answer>B</answer>"
                },
                {
                    "role": "user",
                    "content": answer_text
                }
            ],
            "temperature": 0,
            "max_tokens": 50
        }
        
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"API error for {file_path}: {error_text}")
                return None
                
            result = await response.json()
            answer = result['choices'][0]['message']['content'].strip()
            
            # Get video ID from filename
            video_id = os.path.splitext(os.path.basename(file_path))[0]
            
            # Create output directory if it doesn't exist
            output_dir = "gemini_pro_final_answers"
            os.makedirs(output_dir, exist_ok=True)
            
            # Write result to output file
            output_path = os.path.join(output_dir, f"{video_id}_result.json")
            async with aiofiles.open(output_path, 'w') as f:
                await f.write(json.dumps({"answer": answer}, indent=2))
            
            print(f"Processed {video_id}: {answer}")
            return (video_id, answer)
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

async def process_all_files(directory: str):
    """
    Process all JSON files in the directory concurrently.
    
    Args:
        directory: Directory containing JSON files
    """
    # Get all JSON files in the directory
    json_files = [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.endswith('.json')
    ]
    
    # Configure rate limiting
    semaphore = asyncio.Semaphore(10)  # Limit concurrent API calls
    
    async def process_with_semaphore(file_path: str, session: aiohttp.ClientSession):
        async with semaphore:
            return await process_single_file(file_path, session)
    
    # Process files concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_with_semaphore(file_path, session)
            for file_path in json_files
        ]
        results = await asyncio.gather(*tasks)
    
    # Count successful processes
    successful = len([r for r in results if r is not None])
    print(f"\nProcessed {successful} files successfully")
    print(f"Results saved to {os.path.abspath('gemini_video_final_answers')} directory")

async def main():
    # Directory containing the JSON files
    directory = "gemini_pro_answers"
    
    # Process all files
    await process_all_files(directory)

if __name__ == "__main__":
    load_dotenv()
    # Ensure you have set OPENAI_API_KEY environment variable
    if not os.getenv('OPENAI_API_KEY_KOA_4o'):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
        
    # Run the async main function
    asyncio.run(main())