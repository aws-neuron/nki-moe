import subprocess
import csv


def parse_prompts(filepath):
    with open(filepath, 'r') as file:
        arr = file.read().split('\n\n')
    arr = [prompt.strip() for prompt in arr if prompt.strip()]
    return arr


def parse_prompt_data(filepath):
    with open(filepath, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        rows = list(reader)
    
    return rows

def parse_prompts_as_dict(filepath):
    '''
    Builds a dictionary with the index number of the prompt as the value, the prompt as the key.
    '''
    with open(filepath, 'r') as file:
        arr = file.read().split('\n\n')

    arr = [prompt.strip() for prompt in arr if prompt.strip()]
    rt = {}
    for idx, prompt in enumerate(arr):
        rt[prompt]=idx
    
    return rt


def main():
    prompts = parse_prompts("prompts.txt")
    prompt_data = parse_prompt_data("prompt_data.txt")
    assert len(prompts) == len(prompt_data)

    mode = "evaluate_single"

    # Iterate through the prompts
    for i, prompt in enumerate(prompts):
        data = prompt_data[i]
        seq_len = data[2]
        latency = data[3]
        throughput = data[4]

        command = f'python main.py --mode {mode}'
        
        print(f'Running a test with the following command: {command}')
        
        with open(f'prompt{i}_out.txt', 'w') as outfile:
            subprocess.run(command, shell=True, stdout=outfile)

        print("")


if __name__ == '__main__':
    main()