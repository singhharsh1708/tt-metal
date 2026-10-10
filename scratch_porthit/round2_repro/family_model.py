import numpy as np, torch
T=32; F=16

def tile_pages(a, pad=0):
    """Host tilize of a rank<=2 integer tensor -> list of 1024-word pages (face order)."""
    a=np.atleast_2d(np.asarray(a))
    H,W=a.shape; Hp=-(-H//T)*T; Wp=-(-W//T)*T
    p=np.full((Hp,Wp),pad,dtype=np.int64); p[:H,:W]=a
    pages=[]
    for th in range(Hp//T):
        for tw in range(Wp//T):
            t=p[th*T:(th+1)*T, tw*T:(tw+1)*T]
            pages.append(np.concatenate([t[:16,:16].ravel(),t[:16,16:].ravel(),t[16:,:16].ravel(),t[16:,16:].ravel()]))
    return pages
def rm_pages(a):
    a=np.atleast_2d(np.asarray(a)); return [r.copy() for r in a]
OOB=-777
def rd(pages,page,word):
    if page>=len(pages) or word>=len(pages[page]): return OOB
    return int(pages[page][word])

# ---- A: paged_fill_cache writer: page_table_ptr[virtual_block] from page {batch_idx}
def paged_fill(page_table, layout, batch_idx, n_seq_tiles, block_size_t):
    pages = tile_pages(page_table) if layout=='TILE' else rm_pages(page_table)
    got=[rd(pages,batch_idx,s//block_size_t) for s in range(n_seq_tiles)]
    exp=[int(page_table[batch_idx][s//block_size_t]) for s in range(n_seq_tiles)]
    return got,exp
print("== A/C: page_table lookup (paged_fill_cache writer, ring_distributed_sdpa reader)")
for (B,M,bidx) in [(1,16,0),(1,17,0),(1,32,0),(1,64,0),(4,32,0),(4,32,1),(32,8,3),(1,33,0)]:
    g=torch.Generator().manual_seed(0)
    pt=(torch.randperm(B*M,generator=g)+1).reshape(B,M).numpy()
    for lay in ('ROW_MAJOR','TILE'):
        got,exp=paged_fill(pt,lay,bidx,M,1)
        bad=sum(a!=b for a,b in zip(got,exp)); oob=sum(a==OOB for a in got); zero=sum(a==0 for a in got)
        print(f"  page_table[{B},{M}] batch_idx={bidx} {lay:9s}: wrong_blocks={bad}/{M} (reads_padding_zero={zero}, reads_past_buffer={oob})")

# ---- D: sdpa_decode cur_pos: index_ptr[cur_batch] from page 0
print("== D: sdpa_decode cur_pos_tensor (non-paged)")
for B in (8,16,17,24,32):
    pos=np.arange(100,100+B)
    for lay in ('ROW_MAJOR','TILE'):
        pages = tile_pages(pos) if lay=='TILE' else rm_pages(pos)
        got=[rd(pages,0,b) for b in range(B)]
        bad=[b for b in range(B) if got[b]!=pos[b]]
        print(f"  B={B:2d} {lay:9s}: users_with_wrong_cur_pos={len(bad)} {('first bad user %d reads %d'%(bad[0],got[bad[0]])) if bad else ''}")

# ---- B: moreh_getitem tilized reader, layout define taken from index_tensors.front()
def getitem_idx(idx, layout_of_tensor, define, N):
    pages = tile_pages(np.asarray(idx).reshape(1,-1)) if layout_of_tensor=='TILE' else rm_pages(idx)
    out=[]
    for i in range(N):
        if define=='TILIZE_INDEX':
            page=i//32; t=i%32; word=t if t<16 else t+240
            out.append(rd(pages,page,word))
        else:  # ROW_MAJOR_INDEX: page 0, 32-byte window at floor(i*4/32)*32 -> word i of page 0
            out.append(rd(pages,0,i))
    return out
print("== B: moreh_getitem with mixed index layouts (input TILE, index_dims=[1,2])")
for N in (4,16,32,100):
    g=torch.Generator().manual_seed(2)
    x=torch.randint(0,1000,(10,5,5,64),generator=g)
    i0=torch.randint(0,5,(N,),generator=g); i1=torch.randint(1,5,(N,),generator=g)
    ref=x[:,i0,i1]
    for lays in (('ROW_MAJOR','ROW_MAJOR'),('TILE','TILE'),('ROW_MAJOR','TILE'),('TILE','ROW_MAJOR')):
        define='ROW_MAJOR_INDEX' if lays[0]=='ROW_MAJOR' else 'TILIZE_INDEX'
        r0=getitem_idx(i0.numpy(),lays[0],define,N); r1=getitem_idx(i1.numpy(),lays[1],define,N)
        oob=sum(v==OOB for v in r0+r1)
        badj=sum((a!=int(b)) or (c!=int(d)) for a,b,c,d in zip(r0,i0,r1,i1))
        print(f"  N={N:3d} layouts={lays}: wrong_index_positions={badj}/{N} (reads_past_buffer={oob})")

# ---- F: nonzero on BFLOAT8_B TILE input: tile bytes = 64 exponent bytes + 1024 mantissa bytes
def bfp8_tile(t):  # t: 32x32 float; equal-magnitude nonzeros assumed
    faces=[t[:16,:16],t[:16,16:],t[16:,:16],t[16:,16:]]
    exps=[];man=[]
    for f in faces:
        for r in range(16):
            row=f[r]; mx=np.abs(row).max()
            e=0 if mx==0 else int(np.floor(np.log2(mx)))+127
            exps.append(e)
            for v in row:
                if v==0: man.append(0)
                else:
                    ev=int(np.floor(np.log2(abs(v))))+127; m=int(abs(v)/2.0**(ev-127)*64)>>(e-ev)
                    man.append((0x80 if v<0 else 0)|(m&0x7f))
    return np.array(exps+man,dtype=np.uint8)
def nonzero_kernel_bfp8(x):  # x [H,W] <= 32x32
    H,W=x.shape; t=np.zeros((32,32)); t[:H,:W]=x; b=bfp8_tile(t); out=[]
    for r in range(H):
        for c in range(W):
            m=(((r>>4)<<1)|(c>>4))*256+(r&15)*16+(c&15)
            if b[m]&0x7f: out.append((r,c))
    return out
print("== F: ttnn.nonzero on BFLOAT8_B (TILE) input")
for name,x in [("[1,1,1,32] one-hot at col 5",np.eye(1,32,5)),
               ("[1,1,32,32] identity",np.eye(32)),
               ("[1,1,8,32] ones in row 6",np.pad(np.ones((1,32)),((6,1),(0,0)))),
               ("[1,1,1,32] 2.0 at col 5",2*np.eye(1,32,5))]:
    got=nonzero_kernel_bfp8(x); exp=[tuple(i) for i in np.argwhere(x!=0)]
    print(f"  {name}: torch count={len(exp)} kernel count={len(got)} matching={len(set(got)&set(exp))} kernel_first={got[:4]} torch_first={exp[:4]}")
