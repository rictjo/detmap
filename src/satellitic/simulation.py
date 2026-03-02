lic_ = """
   Copyright 2025 Richard Tjörnhammar

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
from .init import *
# -----------------------
# Top-level Newtonian dynamics
# -----------------------
import numpy as np
from .constants import constants_solar_system, solarsystem
from .constants import celestial_types, build_run_system

TYPE_PLANET     = celestial_types['Planet']
TYPE_STAR       = celestial_types['Star']
TYPE_MOON       = celestial_types['Moon']
TYPE_SATELLITE  = celestial_types['Satellit']
TYPE_OTHER      = celestial_types['Other']

import sys
def excepthook(type, value, tb):
    import traceback
    traceback.print_exception(type, value, tb)
sys.excepthook = excepthook

bUseJax = False
try :
        import jax
        jax.config.update("jax_enable_x64", True)
        bUseJax = True
        print("ImportSuccess:", "HAS JAX IN ENVIRONMENT")
except ImportError :
        print ( "ImportError:","JAX: WILL NOT USE IT")
except OSError:
        print ( "OSError:","JAX: WILL NOT USE IT")

if bUseJax:
    import jax.numpy as xp
else:
    import numpy as xp

def constants( sel = None ) :
    if sel is None :
        return ( constants_solar_system )
    else :
        return ( constants_solar_system[sel] )

def backend_array(x, xp):
    return xp.asarray(x)


"""
def vverlet_a(dt, r, v, a, m , idx_earth=None, idx_leo=None ):
    r1 = r + v*dt + 0.5*a*dt*dt
    a1 = accel(r1, m, idx_earth, idx_leo)
    v1 = v + 0.5*(a + a1)*dt
    return r1, v1, a1

def vverlet_F( dt , r , v, F , m , mm ):
    Fdm	= (F.T / m).T * 0.5
    r1		= r + v*dt + Fdm*dt*dt
    v05	= v + Fdm*dt
    rr		= vdist( r )
    F1	= forceG( rr , mm )
    v1	= v05 + 0.5 * (F1.T / m).T * dt
    return r1 , v1 , F1

def forceG( rr , mm ) :
    G		= constants('G')
    r3		= np.sum(rr**2,axis=2)**1.5
    er3	= rr.T/r3
    er3[np.isnan(er3)] = 0
    return ( np.sum( G * mm * er3 , axis=2).T )

def vdist( r , type=0 ):
    if type==0:
        return ( r[:,None,:] - r[None,:,:] )
    else:
        return ( np.array([[ (r_-w_) for r_ in r] for w_ in r]) )

def productpairs_z( m ):
    return ( np.outer(m,m)*(1-np.eye(len(m))) )

def angular_momentum( r , v , m ):
    return ( np.sum(np.cross(r, m[:,None]*v), axis=0) )

def energy(r, v, m):
    KE = 0.5 * np.sum(m * np.sum(v*v, axis=1))
    rr = vdist(r)
    dist = np.linalg.norm(rr, axis=2)
    np.fill_diagonal(dist, np.inf)
    PE = -0.5 * constants('G') * np.sum(m[:,None]*m[None,:] / dist)
    return KE + PE, KE, PE
"""


if bUseJax :

    @jax.jit
    def accel(r, m, params):

        G  = params["G"]
    
        idx_massive = params["idx_massive"]
        idx_light   = params["idx_light"]

        satellite_indices = params["satellite_indices"]
        satellite_parent  = params["satellite_parent"]

        planet_indices = params["planet_indices"]
        planet_J2 = params["planet_J2"]
        planet_R  = params["planet_R"]
        planet_MU = params["planet_MU"]

        a = xp.zeros_like(r)
		
        # -------------------------
        # Massive ↔ Massive
        # -------------------------
        rM = r[idx_massive]
        mM = m[idx_massive]
        #
        # Massive ↔ Massive GPU SCALABLE O(N^2)
        # PI TODO FMM
        def body_i(ri):
            dr = ri - rM
            r2 = xp.sum(dr * dr, axis=1)
            inv_r3 = xp.where(r2 > 0, r2**(-1.5), 0.0)
            return -G * xp.sum(mM[:, None] * dr * inv_r3[:, None], axis=0)

        aM = jax.vmap(body_i)(rM)

        # PI TODO CONCAT
        # Potential performance issue .at[] inside JIT
        # If preordered according to mass the xp.concatenate
        # will improve performance
        a = a.at[idx_massive].set(aM) 

        # -------------------------
        # Light due to Massive
        # -------------------------
        rL = r[idx_light]

        dr = rL[:, None, :] - rM[None, :, :]
        r2 = xp.sum(dr * dr, axis=2)

        inv_r3 = xp.where(r2 > 0.0, r2**(-1.5), 0.0)

        aL = -G * xp.sum(
            mM[None, :, None] * dr * inv_r3[:, :, None],
            axis=1
        )
        #
        # PI TODO CONCAT
        a = a.at[idx_light].set(aL)

        # -------------------------
        # Vectorized J2
        # -------------------------
        if satellite_indices.size > 0:

            r_planets = r[planet_indices]                     # (P,3)
            r_sats    = r[satellite_indices]                  # (Nsat,3)

            r_parent = r_planets[satellite_parent]            # broadcast
            r_rel    = r_sats - r_parent

            x, y, z = r_rel[:,0], r_rel[:,1], r_rel[:,2]
            r2 = x*x + y*y + z*z
            r5 = r2 * r2 * xp.sqrt(r2)

            J2p = planet_J2[satellite_parent]
            Rp  = planet_R[satellite_parent]
            MUp = planet_MU[satellite_parent]

            factor = 1.5 * J2p * MUp * Rp**2 / r5
            z2_r2 = (z*z)/r2

            ax = factor * x * (5*z2_r2 - 1)
            ay = factor * y * (5*z2_r2 - 1)
            az = factor * z * (5*z2_r2 - 3)

            a_j2 = xp.stack([ax, ay, az], axis=1)
            #
            # PI TODO CONCAT
            a = a.at[satellite_indices].add(a_j2)

        return a

    @jax.jit
    def vverlet( r, v, a, m, params, dt ):
        r1 = r + v*dt + 0.5*a*dt*dt
        a1 = accel(r1, m, params )
        v1 = v + 0.5*(a + a1)*dt
        return r1, v1, a1

    from functools import partial
    @partial(jax.jit, static_argnames=["steps_per_frame"])
    def multi_step(r, v, a, m, params, dt, steps_per_frame ):

        def body(carry, _):
            r, v, a = carry
            r, v, a = vverlet(r, v, a, m, params, dt )
            return (r, v, a), None

        (r, v, a), _ = jax.lax.scan(body, (r, v, a), None, length = steps_per_frame )
        return r, v, a


def accel_np(r, m, params, out=None ):

    G  = params["G"]

    idx_massive = params["idx_massive"]
    idx_light   = params["idx_light"]

    satellite_indices = params["satellite_indices"]
    satellite_parent  = params["satellite_parent"]

    planet_indices = params["planet_indices"]
    planet_J2 = params["planet_J2"]
    planet_R  = params["planet_R"]
    planet_MU = params["planet_MU"]

    if out is None :
        a = np.zeros_like(r)
    else :
        out.fill(0.0)
        a = out

    if out is None :
        # -------------------------
        # Massive ↔ Massive
        # -------------------------
        rM = r[idx_massive]
        mM = m[idx_massive]

        dr = rM[:, None, :] - rM[None, :, :]
        r2 = np.einsum('ijk,ijk->ij', dr, dr)
        np.fill_diagonal(r2, np.inf)

        inv_r = 1.0 / np.sqrt(r2)
        inv_r3 = inv_r * inv_r * inv_r

        aM = -G * np.einsum(
            'ij,ijk,j->ik',
            inv_r3,
            dr,
            mM
        )
        a[idx_massive] = aM

    else :
        # ==========================================================
        # Massive ↔ Massive (symmetric, cache-friendly loops)
        # ==========================================================
        rM = r[idx_massive]
        mM = m[idx_massive]

        NM = rM.shape[0]
        aM = np.zeros_like(rM)

        for i in range(NM - 1):
            ri = rM[i]
            mi = mM[i]

            for j in range(i + 1, NM):
                dr = rM[j] - ri
                r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]

                inv_r = 1.0 / np.sqrt(r2)
                inv_r3 = inv_r * inv_r * inv_r

                f = G * inv_r3

                fx = dr[0] * f
                fy = dr[1] * f
                fz = dr[2] * f

                mj = mM[j]

                # Apply Newton's third law
                aM[i,0] += mj * fx
                aM[i,1] += mj * fy
                aM[i,2] += mj * fz

                aM[j,0] -= mi * fx
                aM[j,1] -= mi * fy
                aM[j,2] -= mi * fz

        a[idx_massive] = aM

    # -------------------------
    # Light due to Massive
    # -------------------------
    rL = r[idx_light]

    dr = rL[:, None, :] - rM[None, :, :]
    r2 = np.sum(dr * dr, axis=2)
    inv_r = 1.0 / np.sqrt(r2)
    inv_r3 = inv_r * inv_r * inv_r

    tmp = dr * inv_r3[:, :, None]
    tmp *= mM[None, :, None]
    aL = -G * tmp.sum(axis=1)
    a[idx_light] = aL

    # -------------------------
    # Vectorized J2
    # -------------------------
    if satellite_indices.size > 0:

        r_planets = r[planet_indices]             # (P,3)
        r_sats    = r[satellite_indices]          # (Nsat,3)

        r_parent = r_planets[satellite_parent]    # (Nsat,3)
        r_rel    = r_sats - r_parent              # (Nsat,3)

        x = r_rel[:, 0]
        y = r_rel[:, 1]
        z = r_rel[:, 2]

        r2 = x*x + y*y + z*z
        r  = np.sqrt(r2)

        inv_r = 1.0 / np.sqrt(r2)
        inv_r5 = inv_r * inv_r * inv_r * inv_r * inv_r

        J2p = planet_J2[satellite_parent]
        Rp  = planet_R [satellite_parent]
        MUp = planet_MU[satellite_parent]

        factor = 1.5 * J2p * MUp * Rp**2 * inv_r5
        z2_r2 = (z*z) * inv_r * inv_r

        ax = factor * x * (5*z2_r2 - 1)
        ay = factor * y * (5*z2_r2 - 1)
        az = factor * z * (5*z2_r2 - 3)

        a[satellite_indices, 0] += ax
        a[satellite_indices, 1] += ay
        a[satellite_indices, 2] += az

    if out is None :
        return a


def vverlet_np( r, v, a, m, params, dt ):
    r1 = r + v*dt + 0.5*a*dt*dt
    a1 = accel_np( r1, m, params )
    v1 = v + 0.5*(a + a1)*dt
    return r1, v1, a1

def multi_step_np(r, v, a, m, params, dt, steps_per_frame ):
    for _ in range(steps_per_frame):
        r, v, a = vverlet_np(r, v, a, m, params, dt )
    return r, v, a


def simulate_small_np(r, v, m, dt,
             steps_per_frame=10,
             params=None):

    if params is None:
        raise ValueError("No parameters")

    # Ensure contiguous memory
    r = np.ascontiguousarray(r)
    v = np.ascontiguousarray(v)
    m = np.ascontiguousarray(m)

    # Preallocate acceleration buffers
    a     = np.zeros_like(r)
    a_buf = np.zeros_like(r)

    # Initial acceleration
    accel_np(r, m, params, out=a)

    step_count = 0

    while True:
        multi_step_inplace(
            r, v, a, m, params,
            dt, steps_per_frame, a_buf
        )

        step_count += steps_per_frame
        yield r, step_count


def vverlet_inplace(r, v, a, m, params, dt, a_buf):
    r += v * dt
    r += 0.5 * a * dt * dt
    # compute new acceleration into buffer
    accel_np(r, m, params, out=a_buf)
    v += 0.5 * (a + a_buf) * dt
    # swap acceleration buffers
    a[:] = a_buf


def multi_step_inplace(r, v, a, m, params, dt, steps_per_frame, a_buf):
    for _ in range(steps_per_frame):
        vverlet_inplace(r, v, a, m, params, dt, a_buf)


def simulate( r, v, m, dt, Nsteps=None, steps_per_frame=10, params=None , bUseJax=bUseJax ):

    if params is None :
        print('Error: No runsystem or jaxed parameters')
        exit(1)

    bSkipRest = False
    if bUseJax:
        import jax.numpy as xp
        accel_fn      = accel
        multi_step_fn = multi_step
    else :
        import numpy as xp
        accel_fn      = accel_np
        multi_step_fn = multi_step_np
        r = np.ascontiguousarray(r)
        v = np.ascontiguousarray(v)
        m = np.ascontiguousarray(m)

        if params['Number of Massive'] < 40 :
            # We dont have JAX and we are small so
            # we are not trying to be vectorized
            # small system handling
            # Preallocate acceleration buffers
            a     = np.zeros_like(r)
            a_buf = np.zeros_like(r)
            multi_step_fn = multi_step_inplace

            bSkipRest = True

            # Initial acceleration
            accel_np(r, m, params, out=a)

            step_count = 0

            while True:
                multi_step_inplace(
                    r, v, a, m, params,
                    dt, steps_per_frame, a_buf
                )

                step_count += steps_per_frame
                yield r, step_count

    if not bSkipRest :
        # We either have JAX or we are large enough
        # to be vectorized
        step_count = 0
        a = accel_fn( r, m, params )

        while True :
            r, v, a = multi_step_fn(r, v, a, m, params, dt,
                steps_per_frame = steps_per_frame )
            step_count += steps_per_frame
            yield r,step_count


def newtonian_simulator( \
    run_parameters      = { 'dt':5e1,
            'Nsteps':None ,
            'steps_per_frame':100 ,
            'mass_epsilon':None ,
            'mass_rule':None } ,
    system_topology     = solarsystem ,
    system_constants    = constants_solar_system ,
    satellite_topology  = None ,
    bAnimated = False ,
    visual_params = { 'visual_moon_scale':10 , 'size_sats':4 },
    bWriteTrajectory = False, trajectory_filename = None ,
    bVerbose = False , bUseJax=bUseJax ) :
    #
    if bUseJax :
        # USER OVERRIDE
        import jax.numpy as xp
    else:
        import numpy as xp

    Nsteps			= run_parameters['Nsteps']
    dt				= run_parameters['dt']   
    steps_per_frame = run_parameters['steps_per_frame']
    
    # Get your steps in
    if not Nsteps is None :
        max_steps = Nsteps * steps_per_frame
    #
    # Setting output names if requested
    if trajectory_filename is not None and bWriteTrajectory == False :
        # The user supplied a name but did not set the flag
        if not '.trj' in trajectory_filename :
            trajectory_filename += '.trj'
        bWriteTrajectory = True
    #
    if bWriteTrajectory == True and trajectory_filename is None :
        # The user wants a timestamped name
        import time
        trajectory_filename = 'traj_' + time.asctime().replace(':','-').replace(' ','_') + '.trj'

    if bVerbose :
        print(f"""Starting simulation of {system_topology}
            using {run_parameters} """)
        if not satellite_topology is None:
            print( f"""with satellite information from :\n {satellite_topology} """)

    run_system = build_run_system( solarsystem,
        constants_solar_system ,
        satellite_topology )
    run_system.apply_barycentric_motion_correction()

    r, v, m, stypes , snames = run_system.phase_state()
    if bVerbose :
        print('Built initial system phase space')
        print( r , '\n' , v , '\n' , m )
    #
    # CREATION AND SETUP OF A LEDGER
    from .constants import InteractionLedger
    if not ( run_parameters['mass_epsilon'] is None and run_parameters['mass_rule'] is None ) :
        run_system.ledger = InteractionLedger( mass_rule = run_parameters['mass_rule'] ,
                    mass_epsilon = run_parameters['mass_epsilon'] )
    else :
        run_system.ledger = InteractionLedger()
    ledger = run_system.ledger
    ledger .constants = constants
    ledger .set_phase_space( run_system.phase_space() )
    if run_system.satellites_object is not None :
        ledger .satellites_objects = [ [sobj[0],*sobj[1].get_index_pairs()] for sobj in run_system.satellites_object ]
    ledger .convert_partition_types(xp)

    const = constants()
    r   = backend_array(r, xp)
    v   = backend_array(v, xp)
    m   = backend_array(m, xp)
    #
    Ncurrent = len(m)
    print( 'Simulating celestial dynamics')
    print( Ncurrent , 'body problem ... ' )

    Nsat = len( xp.where( stypes == celestial_types['Satellit'])[0] )
    if Nsat > 0 :
        print(f'Applied J2 corrections to {Nsat} LEO satellites')

    from .constants import build_params
    params = build_params(run_system)
    if bVerbose:
        print ( 'Jax structs initialized')
        print ( 'Have', params )

    sim = simulate( r, v, m, dt = dt,
            Nsteps = Nsteps, steps_per_frame = steps_per_frame,
            params=params, bUseJax = bUseJax )

    writer = None
    if bWriteTrajectory :
        from .iotools import TrajectoryManager
        writer = TrajectoryManager(
            trajectory_filename,
            particle_types=run_system.get_particle_types(),
            dt_frame=dt*steps_per_frame
        )
        writer.write_cdp(run_system)

    if bAnimated :

        import signal
        def handle_sigint(sig, frame):
            print("\nUser requested shutdown.")
            timer.stop()
            if bWriteTrajectory:
                if writer is not None :
                    writer.close()
            canvas.close()
        #
        # Niceness clash between sigint handlers ...
        signal.signal(signal.SIGINT, handle_sigint)

        #
        # Still need to generalize this
        idx_earth   = run_system.find_indices_of('Earth')[0]
        idx_sun     = run_system.find_indices_of('Sun')[0]
        idx_moon    = run_system.find_indices_of('EarthMoon')[0]
        idx_leo     = run_system.find_indices_of('LEO')

        moon_scale_factor   = visual_params['visual_moon_scale']       # purely visual
        size_sats           = visual_params['size_sats']

        #
        # ---- VisPy 2D projected solar system plot ----
        from vispy import app, scene
        from vispy.scene import visuals
        from vispy.visuals.transforms.linear import MatrixTransform

        # ---- State data container for yields ----
        r_np = np.asarray(r)
        N = r_np.shape[0]
        
        sizes  = np.full(N, 20, dtype=np.float32)
        colors = np.full((N, 4), np.array([0.5, 0.5, 0.5, 1.0]), dtype=np.float32)    

        # special bodies
        sizes[idx_sun]   = 30
        sizes[idx_moon]  = 8
        sizes[idx_leo]   = 2

        colors[idx_sun]   = np.array([1.0, 1.0, 0.0, 1.0])   # yellow
        colors[idx_earth] = np.array([0.0, 0.4, 1.0, 1.0])   # blue
        colors[idx_moon]  = np.array([1.0, 1.0, 1.0, 1.0])   # white

        AU = constants('AU')

        # ---- and visualisation : part 1 ----
        #
        # Canvas & view
        canvas = scene.SceneCanvas(
            keys='interactive',
            size=(800, 800),
            bgcolor='black',
            show=True
        )
        # Whole system
        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.PanZoomCamera(aspect=1)
        view.camera.set_range(
            x=(-1.2*AU, 1.2*AU),
            y=(-1.2*AU, 1.2*AU)
        )

        # Scatter visual
        markers = visuals.Markers()
        markers.set_data(
            r_np[:, :3],
            face_color=colors,
            size=sizes
        )
        view.add(markers)

        # Earth-centric 3D canvas
        canvas_ec = scene.SceneCanvas(
            keys='interactive',
            size=(400, 400),
            bgcolor='black',
            show=True,
            title='Earth-centric view'
        )
        Re = constants('REarth')
        R  = constants('DMoon')  # ~ Earth - Moon distance

        view_ec = canvas_ec.central_widget.add_view()
        viewdistance = Re*2.5
        view_ec.camera = scene.cameras.TurntableCamera(
            fov = 90,
            distance = viewdistance
        )
        f_ = 1.05
        view_ec.camera.set_range(
            x=(-f_*R, f_*R),
            y=(-f_*R, f_*R),
            z=(-f_*R, f_*R)
        )

        earth = visuals.Sphere(
            radius=Re,
            method='latitude',
            color=(0.1, 0.3, 1.0, 0.4)
        )
        view_ec.add(earth)

        # Earth-centric positions
        earth_pos = r_np[idx_earth]
        r_moon = r_np[idx_moon] - earth_pos
        r_sats = r_np[idx_leo]  - earth_pos

        #
        # --- Move Moon ---
        moon = visuals.Sphere(
            radius = constants("RMoon") * moon_scale_factor ,
            method = 'latitude',
            color = (0.8, 0.8, 0.8, 1.0)
        )
        moon.transform = MatrixTransform()
        moon.transform.translate(r_moon)
        view_ec.add(moon)

        # --- Update satellites ---
        sat_marker = visuals.Markers()
        sat_marker.set_data(
            pos  = r_sats,
            size = size_sats,
        )
        view_ec.add(sat_marker)
        
        def update(event) :

            r_new, step_count = next(sim)
            if bUseJax:
                r_new.block_until_ready()
            r_np = np.asarray(r_new)
            #
            # System view
            markers.set_data(
                r_np[:, :3],
                face_color=colors,
                size=sizes
            )
        
            # Earth-centric 3D view
            earth_pos = r_np[idx_earth]
            r_moon = r_np[idx_moon] - earth_pos
            r_sats = r_np[idx_leo] - earth_pos

            # --- Move Moon ---
            moon.transform .reset()
            moon.transform .translate(r_moon)

            sat_marker.set_data(
                pos=r_sats , size = size_sats
            )
            
            if bWriteTrajectory :
                writer.write_step(r_np)

            if Nsteps is not None and step_count >= max_steps:
                print(f"Finished {Nsteps} step simulation")
                timer.stop()
                if bWriteTrajectory:
                    writer.close()
                    canvas.close()
                    return

        timer = app.Timer(interval=1/30)
        timer.connect(update)
        timer.start()
        app.run()

    else :

        print("Running batch simulation...")

        try:
            while True:
                r_new, step_count = next(sim)

                if bUseJax:
                    r_new.block_until_ready()

                if bWriteTrajectory and writer is not None :
                    writer.write_step(np.asarray(r_new))

                if Nsteps is not None and step_count >= max_steps:
                    print(f"Finished {Nsteps} step simulation")
                    break

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")

        finally:
            if write is not None :
                writer.close()
            print("Trajectory file closed cleanly.")


# -----------------------
# Top-level pipeline
# -----------------------
def run_snapshot_simulation(
    out_dir: str = "simulator_output",
    groups: List[str] = CELESTRAK_GROUPS,
    local_tle_file: str = None,
    N_target: int = DEFAULT_N_TARGET,
    grid_nlat: int = DEFAULT_GRID_NLAT,
    grid_nlon: int = DEFAULT_GRID_NLON,
    model: str = DEFAULT_BEAM_MODEL,
    n_beams_per_sat: int = DEFAULT_N_BEAMS_PER_SAT,
    beam_half_angle_deg: float = DEFAULT_BEAM_HALF_ANGLE_DEG,
    beam_pattern: str = DEFAULT_BEAM_PATTERN,
    beam_max_tilt_deg: float = DEFAULT_BEAM_MAX_TILT_DEG,
    beam_gain_model: str = DEFAULT_BEAM_MODEL,
    gain_threshold: float = DEFAULT_GAIN_THRESHOLD,
    frequency_band: str = DEFAULT_FREQUENCY_BAND,
    preferred_bands: Dict[str, Tuple[float,float]] = PREFERRED_BANDS,
    chunk_sat: int = DEFAULT_CHUNK_SAT,
    chunk_ground: int = DEFAULT_CHUNK_GROUND,
    use_gpu_if_available: bool = USE_GPU_IF_AVAILABLE,
    compute_power_map: bool = False,
    save_tles_to_disk: bool = False,
    do_random_sampling:bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    # 1) gather TLEs (CelesTrak primary, local fallback)
    tles = []
    if local_tle_file is None :
        from .iotools import fetch_tle_group_celestrak, parse_tle_text
        for g in groups:
            try:
                print(f"Fetching TLEs for group '{g}' from CelesTrak...")
                raw = fetch_tle_group_celestrak(g)
                tles_group = parse_tle_text(raw)
                print(f"  parsed {len(tles_group)} TLEs from {g}")
                if save_tles_to_disk :
                    fo = open(f"{out_dir+'/'}{g}TLE.txt","w")
                    print ( raw , file=fo )
                    fo.close()
                tles.extend(tles_group)
            except Exception as e:
                print(f"  failed to fetch {g} from CelesTrak: {e}; continuing")

    if len(tles) == 0 :
        from .iotools import load_local_tles
        # local file
        print("No TLEs downloaded from CelesTrak; attempting to load local TLE file:", local_tle_file)
        try:
            tles = load_local_tles(local_tle_file)
            if len(tles) == 0:
                raise RuntimeError("No TLEs available: CelesTrak failed and local file not found/empty.")
        except Exception as e:
            print(f"  failed to obtain tle data from {local_tle_file} : {e}; continuing")
    # Trim to N_target
    if N_target is not None and len(tles) > N_target:
        if do_random_sampling :
            import random
            indices = random.sample( range(len(tles)) , N_target )
            tles = [ tles[ idx ] for idx in indices ]
        else :
            tles = tles[:N_target]
    print("Total TLEs to be used:", len(tles))

    # 2) propagate to epoch
    epoch = datetime.datetime.utcnow()
    print("Propagating TLEs to epoch (UTC):", epoch.isoformat())
    from .propagate import propagate_tles_to_epoch
    names, pos_teme_km, vel_teme_km_s, satrecs = propagate_tles_to_epoch(tles, epoch)
    print("  propagated:", pos_teme_km.shape[0], "satellites")

    # 3) TEME -> ECEF (km)
    from .convert import teme_to_ecef_km, ecef_to_geodetic_wgs84_km
    print("Converting TEME -> ECEF (km) (astropy fallback if available)")
    pos_ecef_km = teme_to_ecef_km(pos_teme_km, epoch)

    # 4) geodetic sub-satellite points (lat/lon/alt)
    lat_s_rad, lon_s_rad, alt_s_km = ecef_to_geodetic_wgs84_km(pos_ecef_km)

    # 5) Build ground grid
    print("Building ground grid (lat/lon)...")
    lat_vals = np.linspace(-60*RAD, 60*RAD, grid_nlat)
    lon_vals = np.linspace(-180*RAD, 180*RAD, grid_nlon)
    lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing='ij')
    ground_lat_flat = lat2d.ravel()
    ground_lon_flat = lon2d.ravel()
    G = ground_lat_flat.size
    print(f"  ground grid: {grid_nlat} x {grid_nlon} = {G} points")

    # 6) decide on GPU usage
    use_gpu = use_gpu_if_available and CUPY_AVAILABLE
    if use_gpu_if_available and not CUPY_AVAILABLE:
        print("CuPy requested but not available; running on CPU (NumPy).")
    if use_gpu:
        print("CuPy detected and will be used for parts of computation (GPU).")

    # 7) aggregate beams to ground
    from .beam import aggregate_beams_to_ground
    print("Aggregating beams to ground (this can be slow for large N; tune chunks)...")
    t0 = time.time()
    total_counts, pref_counts, cofreq_map, power_dbw, Nvis = aggregate_beams_to_ground(
        sat_ecef_km=pos_ecef_km,
        sat_vel_eci_km_s=vel_teme_km_s,
        sat_names=names,
        ground_lat_rad=ground_lat_flat,
        ground_lon_rad=ground_lon_flat,
        model=model,
        n_beams_per_sat=n_beams_per_sat,
        beam_half_angle_deg=beam_half_angle_deg,
        beam_pattern=beam_pattern,
        beam_max_tilt_deg=beam_max_tilt_deg,
        beam_gain_model=beam_gain_model,
        gain_threshold=gain_threshold,
        frequency_band=frequency_band,
        preferred_bands=preferred_bands,
        chunk_sat=chunk_sat,
        chunk_ground=chunk_ground,
        use_gpu=use_gpu,
        compute_power_map=compute_power_map
    )
    t1 = time.time()
    print(f"Aggregation complete in {t1-t0:.1f} s")

    # reshape to 2D for plotting
    total_grid = total_counts.reshape(lat2d.shape)
    pref_grid = pref_counts.reshape(lat2d.shape)
    combined_cofreq_flat = np.zeros_like(total_counts)
    for f, arr in cofreq_map.items():
        combined_cofreq_flat += arr
    combined_cofreq_grid = combined_cofreq_flat.reshape(lat2d.shape)

    # save outputs
    out_total_png = os.path.join(out_dir, "total_beams_heatmap.png")
    out_pref_png = os.path.join(out_dir, "preferred_beams_heatmap.png")
    out_cofreq_png = os.path.join(out_dir, "cofreq_heatmap.png")

    print("Saving heatmaps...")
    from .visualise import plot_heatmap
    plot_heatmap(total_grid, lat_vals, lon_vals, out_total_png, title="Total beams")
    plot_heatmap(pref_grid, lat_vals, lon_vals, out_pref_png, title="Preferred-band beams")
    plot_heatmap(combined_cofreq_grid, lat_vals, lon_vals, out_cofreq_png, title="Co-frequency beams")

    # Save CSVs and grids
    from .convert import save_flat_csv
    out_nvis_csv = os.path.join(out_dir, "nvis_beams.csv")
    out_total_csv = os.path.join(out_dir, "total_beams.csv")
    out_pref_csv = os.path.join(out_dir, "preferred_beams.csv")
    out_cofreq_csv = os.path.join(out_dir, "cofreq_beams.csv")
    save_flat_csv(Nvis, out_nvis_csv, header="nvis_beams")
    save_flat_csv(total_counts, out_total_csv, header="total_beams")
    save_flat_csv(pref_counts, out_pref_csv, header="preferred_beams")
    save_flat_csv(combined_cofreq_flat, out_cofreq_csv, header="cofreq_beams")
    np.save(os.path.join(out_dir, "lat_grid.npy"), lat2d)
    np.save(os.path.join(out_dir, "lon_grid.npy"), lon2d)

    if compute_power_map and (power_dbw is not None):
        out_power_png = os.path.join(out_dir, "received_power_heatmap.png")
        power_grid = power_dbw.reshape(lat2d.shape)
        plot_heatmap(power_grid, lat_vals, lon_vals, out_power_png, title="Received power (dBW)")
        save_flat_csv(power_dbw, os.path.join(out_dir, "received_power.csv"), header="received_power_dBW")

    print("All outputs written to:", out_dir)
    return {
        "total_png"  : out_total_png  ,
        "pref_png"   : out_pref_png   ,
        "cofreq_png" : out_cofreq_png ,
        "total_csv"  : out_total_csv  ,
        "pref_csv"   : out_pref_csv   ,
        "cofreq_csv" : out_cofreq_csv ,
        "nvis_csv"   : out_nvis_csv   
    }

if __name__ == "__main__":
    try:
        out = run_snapshot_simulation(
            out_dir="sim_20251212_dev",
            groups=ALL_CELESTRAK_GROUPS,	# CELESTRAK_GROUPS,
            local_tle_file="tle_local.txt", 	# LOCAL_TLE_FALLBACK,
            N_target=10000,               	# set to 35000 for full-scale runs (ensure resources)
            grid_nlat=120,
            grid_nlon=240,
            model="multibeam",
            n_beams_per_sat=7,
            beam_half_angle_deg=0.8,
            beam_pattern="hex",
            beam_max_tilt_deg=10.0,
            beam_gain_model="gaussian",
            gain_threshold=0.25,
            frequency_band="E-band",
            preferred_bands=PREFERRED_BANDS,
            chunk_sat=256,
            chunk_ground=20000,
            use_gpu_if_available=False,   # set True if you installed cupy
            compute_power_map = True,
            do_random_sampling = True,
        )
        print("Simulation finished. Outputs:", out)
    except Exception as err:
        print("Error during simulation:", err)
        traceback.print_exc()

    import pandas as pd
    tdf = pd.concat( (	pd.read_csv(out['total_csv']),	pd.read_csv(out['pref_csv']),
			pd.read_csv(out['cofreq_csv']),	pd.read_csv(out['nvis_csv'])) )
    print ( tdf .describe() )

    newtonian_simulator ( bAnimated=False,Nsteps=1000,
        tle_file_name	= "/home/rictjo/Downloads/local_tles_smaller.txt" )
    print('Wrote a trajectory file')
    newtonian_simulator ( bAnimated=True,
        tle_file_name	= "/home/rictjo/Downloads/local_tles_smaller.txt" )
