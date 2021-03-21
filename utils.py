import torch

def train(model, opt, loader, device):
    model.train()

    loss_all = 0
    for batch in loader:
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        batch.to(device)
        opt.zero_grad()
        pred,_ = model(batch)
        loss = model.loss(pred, batch.y.type(torch.float32))
        loss.backward()
        opt.step()
        loss_all += batch.num_graphs * loss.item()

        for name, param in model.named_parameters():
            # if 'pan_pool_weight' in name:
            #     param.data = param.data.clamp(0, 1)
            if 'panconv_filter_weight' in name:
                param.data = param.data.clamp(0, 1)
            if 'panpool_filter_weight' in name:
                param.data = param.data.clamp(0, 1)
    return loss_all / len(loader.dataset)


def eval(model, loader, device, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred,_ = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["rocauc"]
