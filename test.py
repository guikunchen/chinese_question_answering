import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast
from tqdm.auto import tqdm
from data.dataloader import get_dataloader, get_test_questions, get_dev_questions


def evaluate(data, output, tokenizer):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    
    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)

        if end_index < start_index:  # prevent from invalid index
            continue
        
        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        
        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            cur_ans = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
            cur_ans = cur_ans.replace(' ','')  # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            if cur_ans != '':  # prevent from empty result
                max_prob = prob
                answer = cur_ans
    
    return answer


def generate_results(model, test_loader, test_questions, device, tokenizer):
    print("Evaluating Test Set ...")

    result = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                        attention_mask=data[2].squeeze(dim=0).to(device))
            result.append(evaluate(data, output, tokenizer))

    result_file = "result.csv"
    with open(result_file, 'w') as f:	
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
            # Replace commas in answers with empty strings (since csv is separated by comma)
            # Answers in kaggle are processed in the same way
                f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

    print(f"Completed! Result is in {result_file}")


def test(model, dataloader, questions, device, tokenizer):
    print("Evaluating Dev Set ...")
    model.eval()
    with torch.no_grad():
        dev_acc = 0
        for i, data in enumerate(tqdm(dataloader)):
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                attention_mask=data[2].squeeze(dim=0).to(device))
            # prediction is correct only if answer text exactly matches
            dev_acc += evaluate(data, output, tokenizer) == questions[i]["answer_text"]
        print(f"Validation | acc = {dev_acc / len(dataloader):.3f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = BertForQuestionAnswering.from_pretrained("checkpoints").to(device)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    _, dev_loader, test_loader = get_dataloader(tokenizer)
    test_questions = get_test_questions()
    dev_questions = get_dev_questions()

    # generate_results(model, test_loader, test_questions, device, tokenizer)
    # test(model, test_loader, test_questions, device, tokenizer)  # test data do not provide the answer
    test(model, dev_loader, dev_questions, device, tokenizer)  # acc = 0.761