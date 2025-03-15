import os
import csv
import asyncio
import logging
from pathlib import Path
import json
from asyncio import Semaphore
from typing import List, Dict
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Part

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiProcessor:
    def __init__(self, project_id: str, location: str = "us-central1", max_concurrent_requests: int = 5):
        """Initialize the Gemini processor"""
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=location
        )
        
        self.model = GenerativeModel("gemini-2.0-flash-001")
        # self.model = GenerativeModel("gemini-1.5-pro")
        self.semaphore = Semaphore(max_concurrent_requests)
        self.request_count = 0
        self.batch_size = 5  # Number of requests before sleeping
        self.sleep_duration = 0  # Sleep duration in seconds

    async def process_question(self, video_id: str, question: str) -> Dict:
        """Process a single question using Gemini API"""
        async with self.semaphore:
            try:
                # Direct GCS path to video
                video_path = f"gs://tesla_videos/videos/{video_id.zfill(5)}.mp4"
                
                # Create video part using GCS path
                video_part = Part.from_uri(
                    uri=video_path,
                    mime_type="video/mp4"
                )
                
                # Prepare prompt
                prompt = f"""You are analyzing a dashcam video taken from the driver's forward-facing perspective.
                
                Using this video, answer the following multiple-choice question by choosing the single best answer.
                Incorporate any relevant details observed in the video (for example, lanes, signage, vehicles, 
                pedestrians, traffic signals, road markings, obstructions) that might help in selecting the correct answer. 
                Explain your reasoning in detail. Relate your findings to each of the multiple-choice options. Eliminate those that are inconsistent with the visual evidence or standard traffic rules, and select the most appropriate remaining choice.
                Conclude with the final choice that best matches the situation. Output that choice in `<answer></answer>` tags.
                
                Question: {question}
                """
                
                # Make API request
                contents = [video_part, prompt]
                response = self.model.generate_content(contents)
                
                # Increment request counter
                self.request_count += 1
                
                # Check if we need to sleep
                if self.request_count % self.batch_size == 0:
                    logger.info(f"Processed {self.request_count} requests. Sleeping for {self.sleep_duration} seconds...")
                    await asyncio.sleep(self.sleep_duration)
                
                return {
                    "video_id": video_id,
                    "answer": response.text
                }

            except Exception as e:
                logger.error(f"Error processing video {video_id}: {str(e)}")
                return {
                    "video_id": video_id,
                    "error": str(e)
                }

    async def process_batch(self, questions: List[Dict]) -> List[Dict]:
        """Process multiple questions in parallel"""
        tasks = []
        for q in questions:
            task = self.process_question(q['id'], q['question'])
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

async def main():
    # Your Google Cloud project ID
    project_id = "tesla-451102"
    
    # Initialize processor
    processor = GeminiProcessor(project_id)
    
    # Create output directory
    output_dir = Path('gemini_video_answers')
    output_dir.mkdir(exist_ok=True)
    
    # Read questions from CSV
    questions = []
    with open('all_questions.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert ID to integer for comparison
            id_num = int(row['id'])
            row['id'] = str(id_num).zfill(5)
            questions.append(row)
    
    # Process questions
    logger.info(f"Processing {len(questions)} questions...")
    results = await processor.process_batch(questions)
    
    # Save results
    for result in results:
        video_id = result.pop('video_id')
        output_path = output_dir / f"{video_id}_result.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved result for video {video_id}")

if __name__ == "__main__":
    asyncio.run(main())