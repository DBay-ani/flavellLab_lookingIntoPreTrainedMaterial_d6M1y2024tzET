# unify all IR models

from models.components.UniFMIR.swinir import *
import copy;
from typing import List;

from torch.nn.functional import interpolate ;


class UniModel(nn.Module):


    @staticmethod
    def bandAver(dim1, dim2):
        #requires(dim1 < dim2);
        bandwidth=dim2-dim1;
        diagSeed=torch.ones(dim2);
        initialTensor=sum([torch.diagflat(diagSeed, offset=x)[\
                int(bandwidth/2):(dim1 + int(bandwidth/2)),:dim2] \
                for x in range(-bandwidth,bandwidth+1)]);
        initialTensor=initialTensor/torch.sum(initialTensor,dim=1).reshape(-1,1);
        return initialTensor



    def __init__(self, \
        img_size, \
        depths, \
        num_heads, \
        inch, \
        n_colors, \
        n_feats, \
        n_resblocks, \
        res_scale, \
        rgb_range, \
        scale, \
        patch_size, \
        embed_dim, \
        window_size, \
        mlp_ratio, \
        qkv_bias, \
        qk_scale, \
        drop_rate, \
        attn_drop_rate, \
        drop_path_rate, \
        norm_layer, \
        patch_norm, \
        use_checkpoint, \
        num_feat, \
        srscale, \
        initialValForFinaleAdditionOf_x2d \
    ) -> None:
        super(UniModel, self).__init__()

        norm_layer=eval(norm_layer);
        self.img_range = 1
        self.mean = torch.zeros(1, 1, 1, 1)
        self.window_size = window_size

        # self.finalAdditionOf_x2d=finalAdditionOf_x2d;
        self.coeffForFinalAddition_x2d = nn.Parameter(torch.ones(1) * initialValForFinaleAdditionOf_x2d);
        self.coeffForFinalAddition_x2d.requires_grad=True;
        self.sigmoidOn_coeffForFinalAddition_x2d = nn.Sigmoid();

        # self.patchDownSamp=nn.Parameter(self.bandAver(50,61));

        # 4 Projection
        specificValOriginalModelHardCodedHere_n_resblocks = 64
        specificValOriginalModelHardCodedHere_n_feats = 256
        ### args.inch = 50
        self.project = Projhead(inch,\
            n_colors,\
            specificValOriginalModelHardCodedHere_n_feats,\
            specificValOriginalModelHardCodedHere_n_resblocks,\
            res_scale,\
            rgb_range,\
            scale);
        self.conv_firstproj = nn.Conv2d(1, embed_dim, 3, 1, 1)
        
        # 5 2D to 3D
        ## Unused ## self.conv_first0 = UNetA(121, 61)
        ## Unused ## self.conv_firstv = nn.Conv2d(61, embed_dim, 3, 1, 1)
        self.conv_before_upsamplev = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.conv_lastv = nn.Conv2d(embed_dim, 61, 3, 1, 1)

        ### self.example12345=nn.Conv2d(embed_dim, 61, 3, 1, 1)
       
        img_size= 8 * patch_size; #  See the line in swinir.py labeled "squirrel", DBayani m27htw17d3M2y2025tzET
        self.patch_embed = PatchEmbed(
            img_size=(img_size), patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        # NOTE, DBayani m29htw18d3M2y2025tzET: it is questionable whether, in the call to PatchUnEmbed below,
        #     the val passed for img_size should be patch_size, of the value assigned to img_size above.
        #     I suppose this is not the only place that question has to be raised.
        self.patch_unembed = PatchUnEmbed(
            img_size=(patch_size), patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers = nn.ModuleList()

        i_layer=0;
        A = dict(      dim=embed_dim,
                         input_resolution=(self.patch_embed.patches_resolution[0],
                                           self.patch_embed.patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection='1conv'
                         )

        for k,v in A.items():
            print(str(k) +", " + str(v), flush=True);

        ### exit();
        for i_layer in range(len(depths)):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(self.patch_embed.patches_resolution[0],
                                           self.patch_embed.patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection='1conv'
                         )
            self.layers.append(layer)
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        # try:
        #     print("_init_weights:" + str(m.name), flush=True);
        # except: 
        #     print("_init_weights:" + str(m)[:1000], flush=True);
        # return;
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(self, x):
       
        # DBayani,m4htw16d15M1y2025tzET, this is the place where things can be put together

        # ~~~~~~~~~~~~ Head ~~~~~~~~~~~~~~~ #
        x2d, closs = self.project(x)
        x2d = self.check_image_size(x2d)
        self.mean = self.mean.type_as(x2d)
        x2d = (x2d - self.mean) * self.img_range
        x = self.conv_firstproj(x2d)

        
        # ~~~~~~~~~~~~ Feature enhancement ~~~~~~~~~~~~~
        xfe = self.conv_after_body(self.forward_features(x))
        ### xfe=x;
        # print("RABBIT:" +str(xfe.shape) + " , " + str(x.shape), flush=True);

        # ~~~~~~~~~~~~ Tail ~~~~~~~~~~~~~~~ #
        # DBayani m9htw16d15M1y2025tzET
        #
        # elif self.task == 4:
        #    x = xfe
        #    x = self.conv_before_upsample0(x)
        #    x = self.conv_last0(x)
        #    return x2d, x / self.img_range + self.mean + x2d  # , closs
        x = xfe
        x = self.conv_before_upsamplev(x)
        x = self.conv_lastv(x)
        rightHandSideToReturn=x / self.img_range + self.mean + \
            self.sigmoidOn_coeffForFinalAddition_x2d(self.coeffForFinalAddition_x2d) * x2d;
        #if(self.finalAdditionOf_x2d):
        #    rightHandSideToReturn=rightHandSideToReturn+x2d;
        ############ rightHandSideToReturn=torch.einsum('ijkl,hj->ihkl',rightHandSideToReturn,self.patchDownSamp);
        sRHS=rightHandSideToReturn.shape;
        # Below, we make a new view since the interpolate function in use expects the first two dimensions to be
        #     batchsize then channels.
        rightHandSideToReturn=interpolate(rightHandSideToReturn.view(sRHS[0],1, sRHS[1], sRHS[2], sRHS[3]), (50,64,64), mode="trilinear")
        return rightHandSideToReturn
        
        #x = x / self.img_range + self.mean
        #
        #return x
    
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

