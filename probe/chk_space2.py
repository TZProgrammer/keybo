from keybo.layout import Layout
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.analysis.surfaces import C30M
from keybo.features import classify as C
L=Layout(''.join(C30M),G)
print('has_key(space):', L.has_key(' '))
print('pos(space):', L.pos(' '), '| hand(0) =', G.hand(0))
print('classify(space,letter):', C.classify_positions(G,(0,0),(-5,3)).value)
print('classify(letter,space):', C.classify_positions(G,(-5,3),(0,0)).value)
print('same_finger(0,-5):', G.same_finger(0,-5))
print('is_lsb(space,letter):', C.is_lsb(G,(0,0),(-2,2)))
print('is_inwards(space,letter):', C.is_inwards(G,(0,0),(-2,2)))
print('is_scissor(space,letter):', C.is_scissor(G,(0,0),(-2,1)))
# and the char->slot mapping direction, verified against the shipped Layout
lay='qwertyuiopasdfghjkl;zxcvbnm,./'.replace(';',"'").replace('/','-')
L2=Layout(lay,G)
ok=all(L2.pos(c)==G.slots[lay.index(c)] for c in lay)
print('char->slot mapping lay.index(c) == Layout.pos(c) for all chars:', ok)
