import torch

class Base_model():
    def __init__(self):
        self.name = ''

    def adjust_lr(self):
        print('Base adjust')
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def say_name(self):
        print('Use adapt model:', self.name)

    def eval_(self):
        self.model.eval()
    
    def save_model(self, path, break_info):
        ckp = {
            'break_info': break_info,
            'model': self.model.state_dict()
        }
        torch.save(ckp, path)

    def load_model(self, path):
        p = torch.load(path, map_location=lambda storage, loc: storage)
        if 'model' in p:
            self.model.load_state_dict(p['model'])
            print('Load model from', path)
            return p['break_info']
        elif 'state_dict' in p:
            self.model.load_state_dict(p['state_dict'])
            print('Load state_dict from', path)

