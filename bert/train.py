import time
import torch

from bert.callbacks import *
from bert.utils import  accuracy_score

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
        logits = output[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        return loss, logits
    else:
        model.eval()

        with torch.no_grad():
            output = model(input_ids,
                           attention_mask=input_masks,
                           labels=labels)
            loss = output[0]
            logits = output[1]
            return loss, logits

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

    callbacks = CallbackList(
        # [DefaultCallback()] + 
        (callbacks or []) +
        [ProgressBarLogger()]
    )
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
    t_start = time.time()

    callbacks.on_train_begin()


    for epoch in range(1, epochs + 1):
        callbacks.on_epoch_begin(epoch)        

        epoch_logs = {}

        model.train()

        for batch_index, batch in enumerate(dataloader):
            # batch_logs = dict(batch=batch_index, size=(batch_size or 1))

            batch_logs = {}
            callbacks.on_batch_begin(batch_index, batch_logs)

            loss, logits = predict_episode(model, batch, optimizer, scheduler)
            
            # batch_logs['loss'] = loss.item()

            callbacks.on_batch_end(batch_index, batch_logs)

            t_elapsed = (time.time() - t_start) / 60
            print(f'Time elapsed: {t_elapsed}')


        callbacks.on_epoch_end(epoch, epoch_logs)

    if verbose:
        print("Finished.")

    callbacks.on_train_end()
