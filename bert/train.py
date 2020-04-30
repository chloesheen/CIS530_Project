import time
import torch

from bert.callbacks import *
from bert.utils import  accuracy_score, format_time

def predict_episode(model,
                    batch,
                    optimizer=None,
                    scheduler=None,
                    train=True):

    device = next(model.parameters()).device
    input_ids = batch[0].to(device)
    input_masks = batch[1].to(device)
    labels = batch[2].to(device)

    if train:
        model.train()
        model.zero_grad()

        output = model(input_ids,
                     attention_mask=input_masks,
                     labels=labels)
        loss = output[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        return loss
    else:
        model.eval()

        with torch.no_grad():
            output = model(input_ids,
                           attention_mask=input_masks)
            logits = output[0]
            return logits

def fit(model,
        optimizer,
        scheduler,
        dataloader,
        epochs=1,
        callbacks=[],
        verbose=True):
    """
    Trains the model
    """
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    loss_values = []

    callbacks = CallbackList([DefaultCallback()] + (callbacks or []) +
                             [ProgressBarLogger()])
    callbacks.set_model(model)
    callbacks.set_params({
        'num_batches': num_batches,
        'batch_size': batch_size,
        'verbose': verbose,
        'optimizer': optimizer,
        'scheduler': scheduler
    })

    if verbose:
        print("Begin training...")

    callbacks.on_train_begin()


    for epoch in range(1, epochs + 1):
        callbacks.on_epoch_begin(epoch)        

        epoch_logs = {}

        model.train()

        for batch_index, batch in enumerate(dataloader):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))

            callbacks.on_batch_begin(batch_index, batch_logs)

            loss = predict_episode(model, batch, optimizer, scheduler)
            
            batch_logs['loss'] = loss.item()

            callbacks.on_batch_end(batch_index, batch_logs)


        callbacks.on_epoch_end(epoch, epoch_logs)

    if verbose:
        print("Finished.")

    callbacks.on_train_end()


def eval_model(model, dataloader):
    """
    Evaluate the model.
    """

    device = next(model.parameters()).device
    t_start = time.time()
    model.eval()

    for step, batch in enumerate(dataloader):
        elapsed = format_time(time.time() - t_start)

        if step % 40 == 0 and not step == 0:
            print("    Batch {:>5,} of {:>5,}.    Elapsed: {:}.".format(step, len(dataloader), elapsed))
    
        
        input_ids = batch[0].to(device)
        input_masks = batch[1].to(device)
        labels = batch[2].to(device)

        with torch.no_grad():

            outputs = model(input_ids, attention_mask=input_masks, labels=labels)

        logits = outputs[0].detach().cpu().numpy()
        labels = labels.detach().cpu().nump()

        batch_accuracy = accuracy_score(logits, labels)

        total_accuracy += batch_accuracy

    total_accuracy = total_accuracy / len(dataloader)
    print("     Accuracy: {0.5f}".format(total_accuracy))
    
    return total_accuracy
