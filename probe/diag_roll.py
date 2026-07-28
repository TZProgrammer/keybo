import sys, numpy as np
from collections import defaultdict
sys.path.insert(0,'/tmp/scissorprice/probe')
import matched_prices as M
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30 as G
def inr(ab): return C.is_inwards(G,*ab)
def outr(ab): return C.is_outwards(G,*ab)
def roll(ab): return inr(ab) or outr(ab)
def shb2(ab): return M.shb(ab)
pairs=[(a,b) for a in M.SLOTS for b in M.SLOTS if a!=b]
print('total ordered distinct pairs:', len(pairs))
n_shb=sum(1 for ab in pairs if shb2(ab)); n_roll=sum(1 for ab in pairs if roll(ab))
n_shb_nonroll=sum(1 for ab in pairs if shb2(ab) and not roll(ab))
n_sfb=sum(1 for ab in pairs if M.sfb(ab))
print(f'same-hand-2finger(shb): {n_shb}   roll(any): {n_roll}   shb & NOT roll: {n_shb_nonroll}   sfb: {n_sfb}')
# is EVERY shb a roll? -> then "shb non-roll" is EMPTY => not identified (trap 16 disjointness)
print('is roll a SUBSET of shb?', all(shb2(ab) for ab in pairs if roll(ab)))
print('is shb a SUBSET of roll?', all(roll(ab) for ab in pairs if shb2(ab)))
# characterise the shb pairs that are NOT rolls
ex=[ab for ab in pairs if shb2(ab) and not roll(ab)]
print(f'{len(ex)} shb-non-roll pairs; examples:', ex[:8])
from collections import Counter
print('their |dx| col-equal counts:', Counter(('samecol' if abs(a[0])==abs(b[0]) else 'diffcol') for a,b in ex))
