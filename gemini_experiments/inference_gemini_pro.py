import os
import csv
from pathlib import Path
import logging
from typing import List, Dict
import json
from google import genai
from google.genai import types
import PIL.Image
from google.generativeai import GenerationConfig
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiFlashProcessor:
    def __init__(self, api_key: str = ""):
        """
        Initialize the Gemini Flash processor with API credentials

        Args:
            api_key (str): Google API key
        """
        self.client = genai.Client(
            api_key=api_key,
            http_options={'api_version': 'v1alpha'}
        )

    def load_frame(self, image_path: str) -> PIL.Image.Image:
        """
        Load an image file using PIL

        Args:
            image_path (str): Path to the image file

        Returns:
            PIL.Image.Image: PIL Image object
        """
        return PIL.Image.open(image_path)

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
        Create the prompt for Gemini

        Args:
            question (str): The question to answer
            frame_count (int): Number of frames

        Returns:
            str: Formatted prompt
        """
        return f"""I am showing you {frame_count} equally spaced frames from a 5-second video. 
The frames are numbered 1 through {frame_count} in chronological order.
Based on these frames, you will answer a multiple choice question. Go through your reasoning through the different frames then put
all that reasoning together to choose the best answer from the multiple choice options. Remember that some frames are more
important than others when it comes to answering the question. Make sure at the end of your answer you 
output the best answer letter choice in <answer></answer> tags. Here is the multiple choice question:

{question}
"""

    def process_question(self, video_id: str, question: str) -> Dict:
        """
        Process a single question with its associated frames

        Args:
            video_id (str): ID of the video/question
            question (str): The question to answer

        Returns:
            Dict: API response
        """
        try:
            # Get frame paths
            frame_paths = self.get_frame_paths(video_id)

            if not frame_paths:
                raise FileNotFoundError(f"No frames found for video ID {video_id}")

            # Load all frames as PIL Images
            frames = [self.load_frame(path) for path in frame_paths]

            # Create contents list with prompt and frames
            contents = [
                self.create_prompt(question),
                *frames
            ]

            # Make API request using the flash-thinking model
            response = self.client.models.generate_content(
                model='gemini-2.0-pro-exp-02-05',
                contents=contents,
                config={
                    "temperature":0.0,
                    "top_p":0.0,
                }
            )

            # Extract the response content
            result = {
                "answer": response.candidates[0].content.parts[0].text if response.candidates else "",
                "candidates": [
                    {
                        "content": candidate.content.parts[0].text,
                        "finish_reason": candidate.finish_reason
                    }
                    for candidate in response.candidates
                ]
            }

            return result

        except Exception as e:
            logger.error(f"Error processing video ID {video_id}: {str(e)}")
            return {"error": str(e)}

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def main():
    # Load API key from environment variable
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    # Initialize processor
    processor = GeminiFlashProcessor(api_key)

    # Create output directory for results
    output_dir = Path('gemini_pro_results')
    output_dir.mkdir(exist_ok=True)

    # Read questions from CSV
    with open('questions.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        # Process each question
        for row in reader:
            video_id = row['id']
            question = row['question']

            logger.info(f"Processing video ID: {video_id}")

            # Process the question
            result = processor.process_question(video_id, question)

            # Save result to JSON file
            output_path = output_dir / f"{video_id}_result.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            logger.info(f"Completed processing video ID: {video_id}")

if __name__ == "__main__":
    main()