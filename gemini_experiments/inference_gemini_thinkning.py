import os
import csv
from pathlib import Path
import logging
from typing import List, Dict
import json
import PIL.Image
from google import genai
from google.generativeai import GenerationConfig
from dotenv import load_dotenv

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
        return f"""You have 5 equally spaced frames (Frame 1 through Frame 5) captured from a 5-second dashcam video, taken from the driver's forward-facing perspective.

Using these frames, answer the following multiple-choice question. Incorporate any relevant details observed in the frames (for example, lanes, signage, vehicles, pedestrians, traffic signals, road markings, obstructions) that might help in selecting the correct answer. Consider how details may change across the frames and note that some frames may be more crucial than others.

Steps to follow:
1. **Frame-by-Frame Analysis:** Briefly describe the significant elements you notice in each of the 5 frames (e.g., signs, road markings, obstructions, other vehicles, potential hazards).
2. **Contextual Reasoning:** Integrate the observations from each frame. Think about what is happening over time, which elements are most relevant, and how they connect to the question.
3. **Match to Answer Choices:** Relate your findings to each of the multiple-choice options. Eliminate those that are inconsistent with the visual evidence or standard traffic rules, and select the most appropriate remaining choice.
4. **Provide the Best Answer:** Conclude with the final choice that best matches the situation. Output that choice in `<answer></answer>` tags.

Now, here is the question and its multiple-choice options:

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

            # Make the API call
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-thinking-exp',
                contents=contents,
                config={
                    "temperature": 0.0,
                    "top_p": 0.0,
                }
            )

            # Extract the response content
            result = {
                "video_id": video_id,
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
            return {
                "video_id": video_id,
                "error": str(e)
            }

    def process_batch(self, questions: List[Dict]) -> List[Dict]:
        """
        Process a batch of questions sequentially

        Args:
            questions (List[Dict]): List of dictionaries containing video_id and question

        Returns:
            List[Dict]: List of results
        """
        results = []
        for q in questions:
            result = self.process_question(q['id'], q['question'])
            
            # Save result immediately after processing each question
            output_dir = Path('gemini_flash_results0')
            output_path = output_dir / f"{q['id']}_result.json"
            
            # Ensure we don't modify the original result when saving
            result_to_save = result.copy()
            result_to_save.pop('video_id', None)  # Remove video_id before saving
            
            with open(output_path, 'w') as f:
                json.dump(result_to_save, f, indent=2)
            
            results.append(result)
            logger.info(f"Processed and saved result for video ID: {q['id']}")
        
        return results

def main():
    # Load environment variables
    load_dotenv()
    
    # Load API key from environment variable
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    # Initialize processor
    processor = GeminiFlashProcessor(api_key)

    # Create output directory for results
    output_dir = Path('gemini_flash_results0')
    output_dir.mkdir(exist_ok=True)

    # Read all questions from CSV
    questions = []
    with open('questions.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        questions = list(reader)

    # Process questions sequentially
    logger.info(f"Processing {len(questions)} questions sequentially...")
    processor.process_batch(questions)  # No need to store results since we're saving as we go
    logger.info("Processing completed!")

if __name__ == "__main__":
    main()