from keybo.layout import Layout
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.analysis.surfaces import C30M
from keybo.features import classify as C
from keybo.data.corpus import load_frequencies, production_corpus_dir
L=Layout(''.join(C30M),G)
print('has_key(space):', L.has_key(' '))
print('pos(space):', L.pos(' '), ' hand:', G.hand(0))
print('classify(space,letter):', C.classify_positions(G,(0,0),(-5,3)))
print('classify(letter,space):', C.classify_positions(G,(-5,3),(0,0)))
CD=production_corpus_dir(None)
for f in ('bigrams.txt','1-skip.txt','trigrams.txt'):
    d=load_frequencies(str(CD/f)); tot=sum(d.values())
    sp=sum(v for k,v in d.items() if ' ' in k)
    print(f'{f}: {len(d)} entries, space-touching mass = {100*sp/tot:.2f}%')
