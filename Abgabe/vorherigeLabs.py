import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__() 
        # Sicherstellen, dass die Output-Dimension durch die Anzahl der Attention Heads teilbar ist, damit jeder Head eine gleich große Dimension hat
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Dimension jedes Attention Heads, berechnet als Gesamt-Output-Dimension geteilt durch die Anzahl der Heads
        self.dropout = nn.Dropout(dropout)
        # Gewichtsmatrizen für die Query-, Key- und Value-Transformationen, jeweils mit einer Linear-Schicht, die die Input-Dimension d_in auf die Output-Dimension d_out transformiert
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) 
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)  # Projektion nach der Multi-Head Attention, um die Ausgabe auf die gewünschte Dimension d_out zu bringen
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        ) # Maskierungsmatrix als Buffer registrieren, um zukünftige Positionen zu maskieren, damit sie nicht als Parameter behandelt wird und nicht aktualisiert wird

    def forward(self, x):
        b, num_tokens, d_in = x.shape # Batchgröße, Anzahl der Tokens im Input, Inputvektorgröße

        keys = self.W_key(x) 
        queries = self.W_query(x)
        values = self.W_value(x)

        # Große Projektionsmatrix (b, num_tokens, d_out) in einzelne Heads aufteilen: (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Achsen tauschen damit jeder Head seine eigene Token-Sequenz zur Verfügung hat: (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Attention Scores für jeden Head berechnen 
        attn_scores = queries @ keys.transpose(2, 3)  

        # Maskierung der Attention Scores, um zukünftige Positionen zu maskieren, damit sie nicht in die Berechnung der Attention Weights einbezogen werden
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Maskierte Attention Scores mit -inf füllen, damit sie nach der Softmax-Operation zu 0 werden 
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # Attention Weights für jeden Head berechnen und Dropout implementieren
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Kontextvektoren für jeden Head berechnen, Ergebnis hat die Form (b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Kontextvektoren von allen Heads zusammenführen, um die finale Ausgabe der Multi-Head Attention zu erhalten, Ergebnis hat die Form (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # Lineare Mischung der zusammengeführten Head-Outputs (optional, üblich aber nicht performance steigernd)

        return context_vec