import os
import torch
import torch.nn as nn
from .attention import ProjectionUpdater
from .Unimodel import UniModel

gpu = torch.cuda.is_available()


class Model(nn.Module):
    def __init__(self, args, ckp, srmodel=None, rp='.'):
        super(Model, self).__init__()
        print('Making model...')
        
        self.scale = args.scale[0]
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu') #  if (not gpu) else 'cuda')
        print(self.device, ' = self.device')
        
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        self.args = args
        
        self.model = srmodel.to(self.device)
        
        self.proj_updater = ProjectionUpdater(self.model, feature_redraw_interval=640)
        if args.precision == 'half':
            self.model.half()
        
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        
        self.dir = os.path.join(rp, 'experiment', args.save)  # , 'model'
        self.load(self.dir, pre_train=args.pre_train, resume=args.resume, cpu=args.cpu)
        print(self.model, file=ckp.log_file)
        print('self.dir = ', self.dir)
        os.makedirs(self.dir, exist_ok=True)
    
    def forward(self, x, tsk):
        self.tsk = tsk
        
        self.proj_updater.redraw_projections()
        return self.model(x, tsk)
    
    def get_model(self):
        return self.model
    
    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)
    
    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_best.pt')
            )
        
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_{}.pt'.format(epoch)))
    
    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        
        if resume == -1:
            print('Load Model from ', os.path.join(apath, 'model_latest.pt'))
            self.model.load_state_dict(
                torch.load(
                    os.path.join(apath, 'model_latest.pt'),
                    **kwargs
                ),
                strict=True
            )
        elif resume == -2:
            m = os.path.join(apath, 'model_best.pt')
            print('Load Model from ', m)
            self.model.load_state_dict(torch.load(m, **kwargs), strict=True)
        elif resume < -2:
            m = os.path.join(apath, 'model_best%d.pt' % -resume)
            print('Load Model from ', m)
            self.model.load_state_dict(torch.load(m, **kwargs), strict=True)
        elif resume == 0 and pre_train != '.':
            print('Loading UNet model from {}'.format(pre_train))
            if os.path.exists(pre_train):
                self.get_model().load_state_dict(torch.load(pre_train, **kwargs), strict=False)
            else:
                print('No pretrain model from {}'.format(pre_train))
                exit()
        elif resume > 0:
            print('Load Model from ', os.path.join(apath, 'model_{}.pt'.format(resume)))
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=True
            )
        else:
            print('!!!!!!!!  Not Load Model  !!!!!!')
            assert resume == 0 and pre_train == '.'
    
    def load_network(self, load_path, strict=True, param_key=None):  # 'params'params_ema
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(load_net, strict=strict)
        print(f'Loading {self.model.__class__.__name__} model from {load_path}.')
    
    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())
        
        if crt_net_keys != load_net_keys:
            print('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                print('warning', f'  {v}')
            print('warning', 'Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                print('warning', f'  {v}')
        
        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print('warning', f'Size different, ignore [{k}]: crt_net: '
                                     f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)
