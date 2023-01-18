"""
Train Multi Layer Perceptron
"""
import data_handler as dh

from mlp import MLP

from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from torch import nn
import torch

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

torch.set_default_dtype(torch.double)


def get_accuracy(predictions, y):
    """Computes the accuracy of a set of predictions"""
    total_correct = 0
    for k in range(predictions.shape[0]):
        pred_label = 1 if predictions[k][0] >= 0.5 else 0
        if pred_label == y[k][0]:
            total_correct += 1

    return total_correct / y.shape[0]


def get_predictions(y_pred):
    """Return a binary predictions 0 for knowledge and 1 for skills"""
    predictions = []
    for pred in y_pred:
        label = 1 if pred[0] >= 0.5 else 0
        predictions.append(label)
    return predictions


def write_to_file(data: list, file_path: str):
    """Save data points to a file"""

    with open(file_path, 'w', encoding='utf-8') as save_file:
        for item in data:
            save_file.write(f'{item}\n')


if __name__ == '__main__':

    LEARNING_RATE = 1e-5
    BATCH_SIZE = 16
    EVAL_STEPS = 2000 * BATCH_SIZE
    N_EPOCHS = 100

    BEST_LOSS_MODEL_SAVE_PATH = './data/models/best_eval_loss_model.pth'
    BEST_ACC_MODEL_SAVE_PATH = './data/models/best_eval_acc_model.pth'
    BEST_AUC_MODEL_SAVE_PATH = './data/models/best_eval_auc_model.pth'

    TRAIN_LOSS_SAVE_PATH = './data/losses/train_losses.txt'
    EVAL_LOSS_SAVE_PATH = './data/losses/eval_losses.txt'
    EVAL_ACC_SAVE_PATH = './data/losses/eval_accuracies.txt'

    MODEL = MLP(batch_size=BATCH_SIZE)
    MODEL.train()

    OPTIM = Adam(MODEL.MLP.parameters(), lr=LEARNING_RATE)
    SCHEDULER = StepLR(OPTIM, step_size=1, gamma=0.75)
    CRITERION = nn.BCELoss()

    BEST_EVAL_LOSS = float('INF')
    BEST_EVAL_ACC = 0
    BEST_EVAL_AUC = 0

    x_train, y_train, x_test, y_test = dh.load_train_test_data(
        './data/train_data/processed_train_data.tsv',
        './data/test_data/processed_test_data.tsv',
    )

    train_loss = 0
    train_losses = []

    eval_losses = []
    eval_accuracies = []
    eval_auc_scores = []

    for epoch in tqdm(range(N_EPOCHS), desc='EPOCH'):
        for i in tqdm(range(0, x_train.shape[0], BATCH_SIZE),
                      desc='BATCH',
                      leave=False
                      ):

            OPTIM.zero_grad()

            x_batch = x_train[i: i + BATCH_SIZE]
            x_batch = torch.from_numpy(x_batch).to(MODEL.device)
            y_batch = y_train[i: i + BATCH_SIZE]
            y_batch = torch.from_numpy(y_batch).to(MODEL.device)
            y_batch = y_batch.unsqueeze(1)

            outputs = MODEL(x_batch).to(MODEL.device)

            loss = CRITERION(outputs, y_batch)
            loss.backward()

            OPTIM.step()

            train_loss += loss.item()

            if i % EVAL_STEPS == 0 and i != 0:
                # EVALUATION LOOP

                MODEL.eval()

                train_losses.append(train_loss / EVAL_STEPS)
                train_loss = 0

                eval_loss = 0
                eval_acc = 0

                eval_labels = []
                eval_predictions = []

                with torch.no_grad():
                    for j in tqdm(range(0, x_test.shape[0], BATCH_SIZE),
                                  desc='EVALUATION',
                                  leave=False
                                  ):
                        x_test_batch = x_test[j: j + BATCH_SIZE]
                        x_test_batch = torch.from_numpy(x_test_batch).to(MODEL.device)
                        y_test_batch = y_test[j: j + BATCH_SIZE]
                        eval_labels += list(y_test_batch)
                        y_test_batch = torch.from_numpy(y_test_batch).double().to(MODEL.device)
                        y_test_batch = y_test_batch.unsqueeze(1)

                        eval_outputs = MODEL(x_test_batch).to(MODEL.device)

                        eval_predictions += get_predictions(eval_outputs)
                        eval_loss += CRITERION(eval_outputs, y_test_batch)

                eval_loss = eval_loss.item() / (x_test.shape[0] / BATCH_SIZE)

                eval_acc = accuracy_score(eval_labels, eval_predictions)
                cm = confusion_matrix(eval_labels, eval_predictions)
                auc_score = roc_auc_score(eval_labels, eval_predictions)

                eval_auc_scores.append(auc_score)
                eval_accuracies.append(eval_acc)
                eval_losses.append(eval_loss)

                if eval_loss < BEST_EVAL_LOSS:
                    torch.save(MODEL, BEST_LOSS_MODEL_SAVE_PATH)
                    BEST_EVAL_LOSS = eval_loss

                if eval_acc > BEST_EVAL_ACC:
                    torch.save(MODEL, BEST_ACC_MODEL_SAVE_PATH)
                    BEST_EVAL_ACC = eval_acc

                if auc_score > BEST_EVAL_AUC:
                    torch.save(MODEL, BEST_AUC_MODEL_SAVE_PATH)
                    BEST_EVAL_AUC = auc_score

                MODEL.train()

    print(f'Best validation acc: {BEST_EVAL_ACC}')
    print(f'Best validation auc: {BEST_EVAL_AUC}')

    write_to_file(train_losses, TRAIN_LOSS_SAVE_PATH)
    write_to_file(eval_losses, EVAL_LOSS_SAVE_PATH)
    write_to_file(eval_accuracies, EVAL_ACC_SAVE_PATH)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(eval_losses, label="Eval Loss")
    plt.xlabel(" Eval. Iteration ")
    plt.ylabel("Loss value")
    plt.legend(loc="upper left")
    plt.show()
    plt.clf()

    plt.plot(eval_auc_scores, label="AUC ROC Score")
    plt.xlabel(" Eval. Iteration ")
    plt.ylabel("AUC ROC")
    plt.legend(loc="upper left")
    plt.show()
    plt.clf()

    plt.plot(eval_accuracies, label="Eval Accuracy")
    plt.xlabel(" Eval. Iteration ")
    plt.ylabel("Accuracy value")
    plt.legend(loc="upper left")
    plt.show()
    plt.clf()
