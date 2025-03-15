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
import PIL.Image
from io import BytesIO

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
        
        self.model = GenerativeModel("gemini-2.0-pro-exp-02-05")
        self.semaphore = Semaphore(max_concurrent_requests)

    def get_frame_paths(self, video_id: str) -> List[str]:
        """Get paths for all frames of a specific video"""
        frame_dir = Path('extracted_frames') / str(video_id)
        frame_paths = sorted(list(frame_dir.glob('frame_*.jpg')))
        return [str(path) for path in frame_paths]

    def image_to_bytes(self, image: PIL.Image.Image) -> bytes:
        """Convert PIL Image to bytes"""
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()

    def create_prompt(self, question: str) -> str:
        """Create the prompt for Gemini"""
        return f"""You have 5 equally spaced frames (Frame 1 through Frame 5) captured from a 5-second dashcam video, taken from the driver’s forward-facing perspective.

Using these frames, answer the following multiple-choice question by choosing the single best answer. Incorporate any relevant details observed in the frames (for example, lanes, signage, vehicles, pedestrians, traffic signals, road markings, obstructions) that might help in selecting the correct answer. Consider how details may change across the frames and note that some frames may be more crucial than others. Explain your reasoning in detail.

Steps to follow:
1. **Frame-by-Frame Analysis:** Describe the significant elements you notice in each of the 5 frames (e.g., signs, road markings, obstructions, other vehicles, potential hazards). Make sure you particularly pay attention to road markings or signs with directional arrows whenever the question asks about the possible directions a given lane can go. 
2. **Contextual Reasoning:** Integrate the observations from each frame. Think about what is happening over time, which elements are most relevant, and how they connect to the question.
3. **Match to Answer Choices:** Relate your findings to each of the multiple-choice options. Eliminate those that are inconsistent with the visual evidence or standard traffic rules, and select the most appropriate remaining choice.
4. **Provide the Best Answer:** Conclude with the final choice that best matches the situation. Output that choice in `<answer></answer>` tags.

Now, here is the question and its multiple-choice options:

{question}
"""

    async def process_question(self, video_id: str, question: str) -> Dict:
        """Process a single question using Gemini API"""
        async with self.semaphore:
            try:
                # Get frame paths
                frame_paths = self.get_frame_paths(video_id)
                
                if not frame_paths:
                    raise FileNotFoundError(f"No frames found for video ID {video_id}")
                
                # Create image parts for each frame
                image_parts = []
                for path in frame_paths:
                    with PIL.Image.open(path) as image:
                        # Convert image to bytes
                        image_bytes = self.image_to_bytes(image)
                        # Create part from bytes
                        image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")
                        image_parts.append(image_part)
                
                # Create prompt
                prompt = self.create_prompt(question)
                
                # Prepare contents list with prompt and frames
                contents = [prompt, *image_parts]
                
                # Make API request with temperature and top_p set to 0
                response = self.model.generate_content(
                    contents,
                    generation_config={
                        "temperature": 0.0,
                        "top_p": 0.0
                    }
                )
                
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
    output_dir = Path('gemini_pro_answers')
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