import os
import csv
import base64
import asyncio
import aiohttp
from openai import AsyncOpenAI
from pathlib import Path
import logging
from typing import List, Dict
import json
from asyncio import Semaphore
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncGPT4VProcessor:
    def __init__(self, api_key: str = "", max_concurrent_requests: int = 5):
        """
        Initialize the GPT-4V processor with API credentials
        
        Args:
            api_key (str): OpenAI API key
            max_concurrent_requests (int): Maximum number of concurrent API requests
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = Semaphore(max_concurrent_requests)

    def encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_frame_paths(self, video_id: str) -> List[str]:
        """
        Get paths for all frames of a specific video
        
        Args:
            video_id (str): ID of the video/question
            
        Returns:
            List[str]: List of frame image paths
        """
        frame_dir = Path('extracted_frames') / str(video_id)
        frame_paths = sorted(list(frame_dir.glob('frame_*.jpg')))
        return [str(path) for path in frame_paths]

    def create_prompt(self, question: str, frame_count: int = 5) -> str:
        """
        Create the prompt for GPT-4V
        
        Args:
            question (str): The question to answer
            frame_count (int): Number of frames
            
        Returns:
            str: Formatted prompt
        """
        # Reasoner models work best with striaght forward prompts and let them do the thinking and reasoning for you. Don't get in thier way!
        return f"""You have 5 equally spaced frames (Frame 1 through Frame 5) captured from a 5-second dashcam video, taken from the driver’s forward-facing perspective.

Using these frames, answer the following multiple-choice question by choosing the single best answer. Conclude with the final choice that best matches the situation. Output that choice in `<answer></answer>` tags.

Now, here is the question and its multiple-choice options:
{question}
"""

    async def process_question(self, video_id: str, question: str) -> Dict:
        """
        Process a single question with its associated frames
        
        Args:
            video_id (str): ID of the video/question
            question (str): The question to answer
            
        Returns:
            Dict: API response
        """
        async with self.semaphore:  # Limit concurrent requests
            try:
                # Get frame paths
                frame_paths = self.get_frame_paths(video_id)
                
                if not frame_paths:
                    raise FileNotFoundError(f"No frames found for video ID {video_id}")

                # Prepare the content list with the initial text prompt
                content = [
                    {
                        "type": "text",
                        "text": self.create_prompt(question)
                    }
                ]
                
                # Add each frame as an image_url
                for path in frame_paths:
                    base64_image = self.encode_image(path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })

                # Make API request
                completion = await self.client.chat.completions.create(
                    model="o1",
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                )
                
                # Extract the response content
                response = {
                    "video_id": video_id,
                    "answer": completion.choices[0].message.content,
                    "finish_reason": completion.choices[0].finish_reason,
                }
                
                return response

            except Exception as e:
                logger.error(f"Error processing video ID {video_id}: {str(e)}")
                return {
                    "video_id": video_id,
                    "error": str(e)
                }

    async def process_batch(self, questions: List[Dict]) -> List[Dict]:
        """
        Process a batch of questions concurrently
        
        Args:
            questions (List[Dict]): List of dictionaries containing video_id and question
            
        Returns:
            List[Dict]: List of results
        """
        tasks = []
        for q in questions:
            task = self.process_question(q['id'], q['question'])
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

async def main():
    # Load environment variables
    load_dotenv()
    
    # Load API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY_KOA_4o')
    if not api_key:
        raise ValueError("OPENAI_API_KEY_KOA_4o environment variable not set")

    # Initialize processor with concurrent request limit
    processor = AsyncGPT4VProcessor(api_key, max_concurrent_requests=5)

    # Create output directory for results
    output_dir = Path('final_answers')
    output_dir.mkdir(exist_ok=True)

    # Read all questions from CSV and add zero padding to IDs
    questions = []
    with open('all_questions.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert ID to int for comparison
            id_num = int(row['id'])
            if 51 <= id_num <= 251:
                # Add zero padding to the ID
                row['id'] = str(id_num).zfill(5)
                questions.append(row)

    # Process questions in parallel
    logger.info(f"Processing {len(questions)} questions in parallel...")
    results = await processor.process_batch(questions)

    # Save results
    for result in results:
        video_id = result.pop('video_id')  # Remove video_id before saving
        output_path = output_dir / f"{video_id}_result.json"
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Completed processing video ID: {video_id}")

if __name__ == "__main__":
    asyncio.run(main())