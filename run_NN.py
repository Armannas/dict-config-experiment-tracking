import numpy as np
import torch
from utils import load_pickle, create_dir, save_pickle

def run_NN(config):
    c_data = config['data']
    c_opt = config['optimizer']
    c_crit = config['criterion']
    c_model = config['model']
    c_trainer = config['trainer']
    c_job = config['job']
    create_dir(f"{c_job['job_dir']}/results")
    res_save_dir = f"{c_job['job_dir']}/results"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = c_data["loader"](**c_data["args"])

    # Instantiate model, loss function and optimizer
    model = c_model["algorithm"](**c_model["args"])
    loss_func = c_crit["algorithm"](**c_crit["args"])
    optimizer = c_opt["algorithm"](model.parameters(), **c_opt["args"])

    train_accs = []
    test_accs = []
    Nbatches = 2 # Want to train fast, this is just for demo purposes
    for epoch in range(1, c_trainer['N_epochs'] + 1):
        print(f'Epoch: {epoch}')
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            if batch_idx <= Nbatches:
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            for batch_idx, (data, target) in enumerate(train_loader, 1):
                if batch_idx <= Nbatches:
                    output = model(data)
                    _, pred = torch.max(output.data, 1)
                    correct += (pred == target).sum().item()
            train_acc = correct / (Nbatches * c_data["args"]["batch_size_train"])
            train_accs.append(train_acc)
            correct = 0
            for batch_idx, (data, target) in enumerate(test_loader, 1):
                if batch_idx <= Nbatches:
                    output = model(data)
                    _, pred = torch.max(output.data, 1)
                    correct += (pred == target).sum().item()
            test_acc = correct / (Nbatches * c_data["args"]["batch_size_test"])
            test_accs.append(test_acc)
        print(f'Train set Accuracy: {train_acc*100:.0f}')
        print(f'Test set Accuracy: {test_acc*100:.0f}')
    torch.save(model.state_dict(), f'{res_save_dir}/model.pth')
    torch.save(optimizer.state_dict(), f'{res_save_dir}/optimizer.pth')

    perf_res = {"train_accs": train_accs,
                "test_accs": test_accs,
                }
    save_pickle(res_save_dir, "perf_res", perf_res)
    # Ran succesfully, save status and updated config
    config['job']['status'] = True
    save_pickle(f"{config['job']['job_dir']}", "config", config)
    return True


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--config_id", help="ID of populated config file of this job")

    ARGS = PARSER.parse_args()
    conf = load_pickle(f"/tmp/config_job{ARGS.config_id}")
    status = run_NN(conf)
