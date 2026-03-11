lic_ = """
   Copyright 2026 Richard Tjörnhammar

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
try :
        import jax
        jax.config.update("jax_enable_x64", True)
        bUseJax = True
except ImportError :
        bUseJax = False
except OSError:
        bUseJax = False


import numpy as np
def strings_find(a, sub, start=0, end=None):
    """
    En ren Python-version av numpy.strings.find.
    
    a   : en itererbar av strängar
    sub : substring att söka efter
    start, end : valfritt sökområde
    """
    results = []
    for s in a:
        # Python slice hanteras direkt av str.find
        idx = str(s).find(sub, start, end)
        results.append(idx)
    return np.array(results)

if bUseJax:
   @jax.jit
    def _split_by_3bits(x):
        x &= 0x1fffff
        x = (x | (x << 32)) & 0x1f00000000ffff
        x = (x | (x << 16)) & 0x1f0000ff0000ff
        x = (x | (x << 8))  & 0x100f00f00f00f00f
        x = (x | (x << 4))  & 0x10c30c30c30c30c3
        x = (x | (x << 2))  & 0x1249249249249249
        return x

    @jax.jit
    def morton3D(x, y, z):
        return (_split_by_3bits(x) |
               (_split_by_3bits(y) << 1) |
               (_split_by_3bits(z) << 2))
else :
    def _split_by_3bits(x):
        x &= 0x1fffff
        x = (x | (x << 32)) & 0x1f00000000ffff
        x = (x | (x << 16)) & 0x1f0000ff0000ff
        x = (x | (x << 8))  & 0x100f00f00f00f00f
        x = (x | (x << 4))  & 0x10c30c30c30c30c3
        x = (x | (x << 2))  & 0x1249249249249249
        return x
    
    def morton3D(x, y, z):
        return (_split_by_3bits(x) |
               (_split_by_3bits(y) << 1) |
               (_split_by_3bits(z) << 2))

if __name__ == '__main__':
    a = ["hello world", "test string", "abcabc", "no match"]
    print ( strings_find(a, "abc") )

    a = np.array(["NumPy is a Python library"])
    print( strings_find(a, "Python") )

    print( 'spam, spam, spam'.find('sp', 1, None) )
