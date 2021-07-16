import torch
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from data.dataloader import get_dataloader, get_dev_questions
from test import evaluate


num_epoch = 30
fp16_training = True
validation = True
logging_step = 100
learning_rate = 1e-4


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if fp16_training:
        from accelerate import Accelerator  # Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
        accelerator = Accelerator(fp16=True)
        device = accelerator.device

    # model
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    train_loader, dev_loader, test_loader = get_dataloader(tokenizer)
    total_steps = len(train_loader) * num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 100, 
                                            num_training_steps = total_steps)

    if fp16_training:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader) 

    # start training
    model.train()

    print("Start Training ...")

    for epoch in range(num_epoch):
        step = 1
        train_loss = train_acc = 0
        
        for data in tqdm(train_loader):	
            # Load all data into GPU
            data = [i.to(device) for i in data]
            
            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

            # Choose the most probable start position / end position
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)
            
            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            train_loss += output.loss
            
            if fp16_training:
                accelerator.backward(output.loss)
            else:
                output.loss.backward()
            
            optimizer.step()
            scheduler.step()  # after an epoch finished
            optimizer.zero_grad()
            step += 1
            ##### TODO: Apply linear learning rate decay #####
            
            # Print training loss and accuracy over past logging step
            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0
        
        # scheduler.step()  # after an epoch finished

        if validation:
            dev_questions = get_dev_questions()
            print("Evaluating Dev Set ...")
            model.eval()
            with torch.no_grad():
                dev_acc = 0
                for i, data in enumerate(tqdm(dev_loader)):
                    output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                        attention_mask=data[2].squeeze(dim=0).to(device))
                    # prediction is correct only if answer text exactly matches
                    dev_acc += evaluate(data, output, tokenizer) == dev_questions[i]["answer_text"]
                print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
            model.train()

    # Save a model and its configuration file to the directory 「checkpoints」 
    # i.e. there are two files under the direcory 「checkpoints」: 「pytorch_model.bin」 and 「config.json」
    # Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("checkpoints")」
    print("Saving Model ...")
    model_save_dir = "checkpoints" 
    model.save_pretrained(model_save_dir)


if __name__ == "__main__":
    train()


""" Task description
- Chinese Extractive Question Answering
  - Input: Paragraph + Question
  - Output: Answer

- Objective: Learn how to fine tune a pretrained model on downstream task using transformers

- Todo
    - Fine tune a pretrained chinese BERT model
    - Change hyperparameters (e.g. doc_stride)
    - Apply linear learning rate decay
    - Try other pretrained models
    - Improve preprocessing
    - Improve postprocessing
- Training tips
    - Automatic mixed precision
    - Gradient accumulation
    - Ensemble

- Estimated training time (tesla t4 with automatic mixed precision enabled)
    - Simple: 8mins
    - Medium: 8mins
    - Strong: 25mins
    - Boss: 2hrs
"""
