import torch
import torch.nn as nn


class Patch_encodage(nn.Module):
    def __init__(self,img_size,patch_size,in_channels,embedding_dim):
        super().__init__()
        assert img_size%patch_size==0 #On doit s'assurer de bien pouvoir diviser l'image
        self.patch_size = patch_size
        self.num_patches = (img_size//patch_size)**2 
        #On fait deux étapes en une: on projette linéairement
        self.linearProj  = nn.Conv2d(in_channels,embedding_dim,kernel_size=patch_size,stride = patch_size)
        
    def forward(self,x):
        #[B,C,H,W]
        x = self.linearProj(x)
        #[B,embedding_dim,H//patch_size,W//patch_size]
        x = x.flatten(2)
        #[B,embedding_dim,num_patches]
        x = x.transpose(1,2)
        #[B,num_patches,embedding_dim]
        return x
        
class MLP(nn.Module):
    def __init__(self,in_dim,hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim,hidden_dim)
        self.ac1=nn.GELU()
        self.fc2 = nn.Linear(hidden_dim,in_dim)
    def forward(self,x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        assert embed_dim%num_heads ==0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads
        self.scale = 1/self.head_dim**0.5
        #Projection pour Q, K et V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        #Projection après concaténation
        self.out_proj = nn.Linear(embed_dim,embed_dim)
        
        
    def forward(self,x):
        B,N,E = x.shape
        #On est au départ de la forme [Batches, nombre de patches, dimension embedding]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        #Decoupage en plusieurs têtes
        Q = Q.view(B,N,self.num_heads,self.head_dim) 
        # [B,N,number_heads,head_dim]
        Q = Q.transpose(1,2)
        # [B, number_heads, N, head_dim]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        #Calcul du score d'attention
        attention_score = Q@K.transpose(-2,-1)
        attention_score = attention_score * self.scale
        # [B,number_heads, N,N]
        
        #Calcul de softmax
        attention_score = attention_score.softmax(dim=-1)
        
        #Calcul final
        out = attention_score@V
        
        #Concaténation des têtes
        out = out.transpose(1,2).reshape(B,N,E)
        
        #projection linéaire finale
        out = self.out_proj(out)
        return out
    
    
class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Creation du Norm
        self.norm1 = nn.LayerNorm(embed_dim)
        
        #MHA
        self.mha = MultiHeadAttention(embed_dim,num_heads)
        
        #Norm avant MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        
        #MLP
        hidden_dim = int(embed_dim*4)
        self.mlp = MLP(embed_dim,hidden_dim)
        
    def forward(self,x):
        residu = x.clone()
        x = self.norm1(x)
        x = self.mha(x)
        x=x+residu
        residu = x.clone()
        x = self.norm2(x)
        x=self.mlp(x)
        x=x+residu
        return x
    
    
class VIT(nn.Module):
    def __init__(self,img_size=28,patch_size=7,in_channel=1
                 ,embed_dim=64,num_heads=4,depth=6,num_classes=10):
        super().__init__()
        
        #Encodage des patch
        self.patch_embedding = Patch_encodage(img_size,patch_size,in_channel,embed_dim)
        num_patches = (img_size//patch_size)**2
        
        #Token CLS et position embedding
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+1,embed_dim))
        
        #Bloc transformer
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])

        #Normalsiation finale
        self.norm = nn.LayerNorm(embed_dim)
        
        #Tête de classification
        self.head = nn.Linear(embed_dim,num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self,x):
        B=x.shape[0]
        #patch embedding
        x = self.patch_embedding(x)
        #Ajout du token cls
        cls_tokens = self.cls_token.expand(B,-1,-1) # [1,1,embed_dim] ==> [B,1,embed_dim]
        x= torch.cat((cls_tokens,x),dim=1) #[B,N+1,E]
        
        #Ajout des positions embedding
        x=x+self.pos_embed
        
        #transformeur
        for block in self.blocks:
            x = block(x)

        #normalisation finale
        x = self.norm(x)
        cls_tokens = x[:,0] #[B,E]
        logits = self.head(cls_tokens) #[B,nb_classes]
        return logits