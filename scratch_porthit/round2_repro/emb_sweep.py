from emb_model import model
def pc(vol,cores,al=16):
    u=al; req=(vol+u-1)//u
    if req>cores: u=((vol+cores-1)//cores+al-1)//al*al
    return u
print("CI shapes (expect 0 bad):")
for B,S in [(1,32),(8,32),(9,32),(1,256),(8,256),(9,256),(1,512),(8,512),(9,512),(2,640)]:
    print((B,S),[ (c,model(B,S,pc(B*S,c))[0]) for c in (56,64)])
print("sweep:")
rows=[]
for B in [16,17,20,24,32,33,40,48,64,96,128]:
    for S in [16,20,32,33,40,48,50,64,77,96,100,128,256,512]:
        r=[]
        for c in (56,64):
            nb,vol=model(B,S,pc(B*S,c)); r.append((c,pc(B*S,c),nb))
        if any(x[2] for x in r): rows.append(((B,S),B*S,r))
for x in rows: print(x)
