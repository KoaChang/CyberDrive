import pandas as pd
from collections import Counter

def find_majority_answers(csv_files):
    """
    Find the majority answer for each ID across multiple CSV files.
    
    Args:
        csv_files (list): List of CSV filenames to process
        
    Returns:
        pandas.DataFrame: DataFrame with ID and majority answer
    """
    # Dictionary to store answers for each ID
    all_answers = {}
    
    # Process each CSV file
    for file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Ensure required columns exist
            if 'id' not in df.columns or 'answer' not in df.columns:
                print(f"Error: Required columns missing in {file}")
                continue
                
            # Process each row
            for _, row in df.iterrows():
                id_num = row['id']
                answer = row['answer']
                
                # Initialize list for this ID if it doesn't exist
                if id_num not in all_answers:
                    all_answers[id_num] = []
                    
                # Add this answer to the list for this ID
                all_answers[id_num].append(answer)
                
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    # Calculate majority answer for each ID
    majority_results = []
    for id_num, answers in all_answers.items():
        # Find the most common answer
        if answers:
            # Count occurrences of each answer
            counter = Counter(answers)
            max_count = max(counter.values())
            
            # Get all answers that appear the maximum number of times
            max_answers = [ans for ans, count in counter.items() if count == max_count]
            
            # If there's a tie, use the alphabetically first answer
            majority_answer = min(max_answers)
            
            majority_results.append({
                'id': id_num,
                'answer': majority_answer
            })
    
    # Convert results to DataFrame and sort by ID
    result_df = pd.DataFrame(majority_results)
    result_df = result_df.sort_values('id')
    
    return result_df

def main():
    # List of CSV files to process
    csv_files = [
        'gemini_pro1.csv',
        'gemini_pro2.csv',
        'open_ai1.csv',
        'open_ai2.csv',
        'open_ai3.csv'
    ]
    
    # Find majority answers
    result_df = find_majority_answers(csv_files)
    
    # Save results to CSV
    output_file = 'majority_results.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()