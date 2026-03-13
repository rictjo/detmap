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
bUseJax = False
try :
        import jax
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

if bUseJax :
    import jax
    import jax.numpy as jnp
    from jax import lax
    #
    # ============================================================
    # Morton curve methods
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

    @jax.jit
    def compute_cells(r, cell_size):
        cells = jnp.floor(r / cell_size).astype(jnp.int32)
        # shift so negative coordinates work
        cells = cells - jnp.min(cells, axis=0)
        return cells

    @jax.jit
    def compute_morton_codes(r, cell_size):
        cells = compute_cells(r, cell_size)
        codes = morton3D(
            cells[:,0].astype(jnp.uint64),
            cells[:,1].astype(jnp.uint64),
            cells[:,2].astype(jnp.uint64)
        )
        return cells, codes
    #
    # ============================================================
    # Hilbert curve methods
    @jax.jit
    def hilbert3D(x, y, z, bits=21):
        """
    Fast 3D Hilbert index.

    x,y,z : uint64 coordinates
    bits  : number of bits per dimension (<=21)

    returns uint64 Hilbert index
        """
        x = x.astype(jnp.uint64)
        y = y.astype(jnp.uint64)
        z = z.astype(jnp.uint64)
        mask = jnp.uint64(1) << (bits - 1)

        def body(i, state):
            x, y, z, h, mask = state
            xi = (x & mask) > 0
            yi = (y & mask) > 0
            zi = (z & mask) > 0

            digit = (
                (xi.astype(jnp.uint64) << 2) |
                (yi.astype(jnp.uint64) << 1) |
                zi.astype(jnp.uint64)
            )
            h = (h << 3) | digit

            # rotation / reflection step
            swap_xy = (~zi) & (xi ^ yi)
            swap_xz = (~yi) & (xi ^ zi)

            x2 = jnp.where(swap_xy, y, x)
            y2 = jnp.where(swap_xy, x, y)
            x3 = jnp.where(swap_xz, z, x2)
            z2 = jnp.where(swap_xz, x2, z)
            mask = mask >> 1
            return (x3, y2, z2, h, mask)

        x, y, z, h, mask = lax.fori_loop(
            0,
            bits,
            body,
            (x, y, z, jnp.uint64(0), mask)
        )
        return h
   
    @jax.jit
    def hilbert3D_vec(x, y, z, bits=21):
        return jax.vmap(lambda a, b, c: hilbert3D(a, b, c, bits))(x, y, z)

    @jax.jit
    def compute_hilbert_codes(r, cell_size, bits=21):
        cells = jnp.floor(r / cell_size).astype(jnp.int32)
        # shift so negatives work
        cells = cells - jnp.min(cells, axis=0)
        x = cells[:,0].astype(jnp.uint64)
        y = cells[:,1].astype(jnp.uint64)
        z = cells[:,2].astype(jnp.uint64)
        codes = hilbert3D_vec(x, y, z, bits)
        return cells, codes

    # ============================================================
    # Build cell start/end lookup
    #
    @jax.jit
    def build_cell_table(codes):
        order = jnp.argsort(codes)
        codes_sorted = codes[order]
        unique, start = jnp.unique(codes_sorted, return_index=True)
        end = jnp.concatenate([start[1:], jnp.array([codes_sorted.shape[0]])])
        return order, codes_sorted, unique, start, end

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

    def compute_cells(r, cell_size):
        cells = np.floor(r / cell_size).astype(jnp.int32)
        # shift so negative coordinates work
        cells = cells - np.min(cells, axis=0)
        return cells

    def compute_morton_codes(r, cell_size):
        cells = compute_cells(r, cell_size)
        codes = morton3D(
            cells[:,0].astype(np.uint64),
            cells[:,1].astype(np.uint64),
            cells[:,2].astype(np.uint64)
        )
        return cells, codes

    # ============================================================
    # Build cell start/end lookup
    #
    def build_cell_table(codes):
        order = np.argsort(codes)
        codes_sorted = codes[order]
        unique, start = np.unique(codes_sorted, return_index=True)
        end = np.concatenate([start[1:], jnp.array([codes_sorted.shape[0]])])
        return order, codes_sorted, unique, start, end


if __name__ == '__main__':
    a = ["hello world", "test string", "abcabc", "no match"]
    print ( strings_find(a, "abc") )

    a = np.array(["NumPy is a Python library"])
    print( strings_find(a, "Python") )

    print( 'spam, spam, spam'.find('sp', 1, None) )
