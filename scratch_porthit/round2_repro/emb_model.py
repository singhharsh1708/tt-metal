import numpy as np, itertools, sys
F=16;T=32;FHW=256
def tilize2d(a):
    B,S=a.shape; Hp=-(-B//T)*T; Wp=-(-S//T)*T
    p=np.zeros((Hp,Wp),dtype=np.int64); p[:B,:S]=a
    tiles=[]
    for th in range(Hp//T):
        for tw in range(Wp//T):
            t=p[th*T:(th+1)*T, tw*T:(tw+1)*T]
            faces=[t[:16,:16],t[:16,16:],t[16:,:16],t[16:,16:]]
            tiles.append(np.concatenate([f.reshape(-1) for f in faces]))
    return tiles
def run_core(tiles,row_length,tile_offset,face_offset,num_rows,curr_col,starting_index,one_face_col):
    out=[]
    curr_tile=tile_offset; offset=face_offset; index=starting_index; read=True; col_offset=curr_col
    tiles_per_row=(row_length+T-1)//T
    buf=None
    for i in range(num_rows):
        if read:
            buf=tiles[curr_tile] if 0<=curr_tile<len(tiles) else None; read=False
        pos=index+offset
        out.append(int(buf[pos]) if (buf is not None and 0<=pos<1024) else -999)
        index+=1; col_offset+=1
        if index==F or col_offset==row_length:
            index=0
            face=offset//FHW
            if col_offset==row_length:
                read=True; col_offset=0
                if offset==T*T:
                    curr_tile+=1; offset=0
                else:
                    curr_tile-=(tiles_per_row-1)
                    if face%2==0:
                        if one_face_col:
                            if face==0:
                                l=(offset-(FHW-F)) & 0xffffffff
                                if l<F: offset+=FHW
                                offset+=F
                            else:
                                l=(offset-(FHW*3-F)) & 0xffffffff
                                if l<F:
                                    offset=0; curr_tile+=1
                                else: offset+=F
                        else:
                            offset+=F
                    else:
                        offset-=F*(F-1)
            elif face%2==0:
                offset+=FHW
            else:
                curr_tile+=1; offset-=FHW; read=True
    return out
def model(B,S,per_core):
    a=np.arange(1,B*S+1).reshape(B,S)
    tiles=tilize2d(a)
    vol=B*S; res=[]; w=0
    tprow=(S+T-1)//T
    while w<vol:
        n=min(per_core,vol-w)
        col=w%S; row=w//S
        r_f=(((row%T)//F)*2*FHW)+((row%F)*F)
        c_f=((col%T)//F)*FHW
        res+=run_core(tiles,S,((row//T)*tprow)+(col//T),r_f+c_f,n,col,col%F,S<=F)
        w+=n
    return int((np.array(res)!=a.reshape(-1)).sum()),vol
if __name__=='__main__':
    bad={}
    for B in [1,2,3,8,15,16,17,31,32,33,40,64]:
        for S in [1,5,15,16,17,24,31,32,33,48,50,64,96,100,128]:
            for pc in [10**9,16,32,48,64,128]:
                nb,vol=model(B,S,pc)
                if nb: bad.setdefault((B,S),[]).append((pc,nb,vol))
    for k,v in sorted(bad.items()): print(k,v)
    print('total bad shapes',len(bad))
