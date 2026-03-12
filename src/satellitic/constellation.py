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
"""
Generate scalable Walker-style satellite constellations and export TLEs
from pandas DataFrame definitions.
"""
from .init import *

import numpy as np
import pandas as pd
from datetime import datetime

MU = MU_EARTH_GRAV
R_EARTH = R_EARTH_KM

# ---------------------------------------------------------------------
# Orbital utilities
# ---------------------------------------------------------------------

def mean_motion_rev_per_day(alt_km: float) -> float:
    """
    Convert circular-orbit altitude to mean motion (rev/day).
    """
    a = R_EARTH + alt_km
    n_rad_s = np.sqrt(MU / a**3)
    return n_rad_s * 86400.0 / (2.0 * np.pi)

# ---------------------------------------------------------------------
# TLE generation
# ---------------------------------------------------------------------
def tle_checksum(line: str) -> int:
    cksum = 0
    for c in line:
        if c.isdigit():
            cksum += int(c)
        elif c == '-':
            cksum += 1
    return cksum % 10

def format_tle(
    satnum,
    epoch,
    inc,
    raan,
    ecc,
    argp,
    M,
    n,
    bstar=0.0,
    revnum=1
):
    year = epoch.year % 100
    doy = (epoch - datetime(epoch.year, 1, 1)).total_seconds() / 86400 + 1
    epoch_str = f"{year:02d}{doy:012.8f}"

    ecc_str = f"{ecc:.7f}".split('.')[1]

    bstar_str = f"{bstar:.5e}".replace('e', '').replace('+', '')

    l1 = (
        f"1 {satnum:05d}U 00000A   {epoch_str}  "
        f".00000000  00000-0 {bstar_str:>8} 0  999"
    )
    l1 += str(tle_checksum(l1))

    l2 = (
        f"2 {satnum:05d} "
        f"{inc:8.4f} {raan:8.4f} {ecc_str:7s} "
        f"{argp:8.4f} {M:8.4f} {n:11.8f}{revnum:5d}"
    )
    l2 += str(tle_checksum(l2))

    return l1, l2

def generate_constellation_tles(
    df: pd.DataFrame,
    satnum_start: int = 10000,
    eccentricity: float = 1e-4,
    argp_deg: float = 0.0,
    bstar: float = 0.0
) -> pd.DataFrame:
    """
    Generate TLEs for multiple constellation systems defined in a DataFrame.

    Required DataFrame columns:
        system
        height_km
        n_planes
        sats_per_plane
        inclination_deg
        raan0_deg

    Returns:
        DataFrame with one row per satellite:
            system, satnum, plane, slot, tle1, tle2
    """

    tles = []
    satnum = satnum_start
    epoch = datetime.utcnow()

    # Epoch in TLE fractional day-of-year format
    epoch_day = (
        (epoch - datetime(epoch.year, 1, 1)).days + 1
        + (epoch.hour + epoch.minute / 60 + epoch.second / 3600) / 24
    )

    for _, row in df.iterrows():
        n_planes = int(row.n_planes)
        sats_per_plane = int(row.sats_per_plane)

        mean_motion = mean_motion_rev_per_day(row.height_km)
        mean_motion_rad_min = mean_motion * 2.0 * np.pi / 1440.0

        for p in range(n_planes):
            raan_deg = (row.raan0_deg + p * 360.0 / n_planes) % 360.0

            for s in range(sats_per_plane):
                mean_anomaly_deg = (s * 360.0 / sats_per_plane) % 360.0

                tle1, tle2 = format_tle(
                                satnum=satnum,
                                epoch=epoch,
                                inc=row.inclination_deg,
                                raan=raan_deg,
                                ecc=eccentricity,
                                argp=argp_deg,
                                M=mean_anomaly_deg,
                                n=mean_motion,
                                bstar=bstar
                )

                tles.append({
                    "system": row.system,
                    "satnum": satnum,
                    "plane": p,
                    "slot": s,
                    "tle1": tle1,
                    "tle2": tle2
                })

                satnum += 1

    return pd.DataFrame(tles)

def repack_input( selection:list[str] , study_systems:list ) -> list :
    data = []
    for sel,mdata in zip(selection,study_systems):
        for row in mdata :
            data.append([sel,*row])
    return ( data )

def build_constellation_df( input:list ) -> pd.DataFrame :
    return ( pd.DataFrame( input ,
      columns = [
        "system" ,
        "height_km" ,
        "n_planes" ,
        "sats_per_plane" ,
        "inclination_deg" ,
        "raan0_deg"
      ]
    ) )

def create_tle_from_system_selection( selection , systems_information = systems_5Cs142dE_20241108 ,
					system_names = recommended_system_names , bVerbose=False ,
					output_file = None ) :

    study_systems	= [ systems_information[sys]	for sys in selection ]

    constellation_df = build_constellation_df ( repack_input( selection , study_systems ) )

    if bVerbose:
        print ( constellation_df )
        print ( "Generating TLEs..." )

    tle_df = generate_constellation_tles(
        constellation_df ,
        satnum_start = 10000
    )

    print ( f"Generated {len(tle_df)} satellites\n" )

    # Swap in system names
    tle_df['system'] = [  v + '-' + system_names[v] for v in tle_df.loc[:,'system'].values ]

    if bVerbose :
        print ( tle_df )

    # Optional: write to file
    if output_file is not None :
        with open(output_file, "w") as f:
            for _, row in tle_df.iterrows():
                f.write(row.tle1 + "\n")
                f.write(row.tle2 + "\n")

    return tle_df


recommended_system_names = {'H' : 'Oneweb', 'A' : 'SpaceX', 'B' : 'Kuiper', 'D' : 'Telesat', 'I' : 'SES Astra LEO' }
systems_5Cs142dE_20241108 = {'A':[[525,28,120,53,0],
		[530,28,120,43,0],
		[535,24, 28,33,0],
		[535, 4, 27,33,0]],
           'B':[[590,28, 28,33,0],
		[610,36, 36,42,0],
		[630,34, 34,51.9,0]],
           'C':[[35786,1,1,0,0]],
           'D':[[1050,12,28,89,0]],
           'E':[[1414,8,6,52,0]],
           'F':[[range(450,900+1),81,range(1,8+1),[r*0.1 for r in range(990)],0]],
           'G':[[range(340,614+1),794,range(12,120+1),range(33,148+1),0]],
           'H':[[[600,1200],132,range(36,72+1),[i*0.1 for i in range(400,880)],0]],
           'I':[[8062,1,32,0,0],
		[8062,4,16,90,0],
		[8062,6,12,45,0]],
           'J':[[1175,18,48,86.5,0]],
           'K':[[355,24,124,50,0],
		[347,24,124,50.2,0]],
           'L':[[*a,0] for a in zip([500, 500, 600, 600, 700, 700, 800, 800, 900, 900, 1000, 1000, 1100, 1100, 1200, 1200, 1300, 1300, 1400, 1400, 8100, 8100, 8100, 8100, 8100, 8100, 8100, 8100, 12000, 12000, 12000, 12000, 12000, 12000, 12000, 12000, 16000, 16000, 16000, 16000, 16000, 16000, 16000, 16000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 23222, 23222] , [36, 36, 36, 36, 34, 34, 30, 30, 28, 28, 24, 24, 24, 24, 22, 24, 20, 20, 18, 18, 1, 12, 12, 12, 12, 12, 12, 12, 1, 12, 12, 12, 12, 12, 12, 12, 1, 12, 12, 12, 12, 12, 12, 12, 1, 12, 12, 12, 12, 12, 12, 12, 1, 12] , [36, 36, 32, 32, 32, 32, 32, 32, 30, 30, 24, 24, 24, 24, 24, 24, 24, 24, 20, 20, 96, 10, 10, 10, 10, 10, 10, 10, 96, 10, 10, 10, 10, 10, 10, 10, 96, 10, 10, 10, 10, 10, 10, 10, 96, 10, 10, 10, 10, 10, 10, 10, 96, 9] , [50, 85, 50, 85, 50, 85, 50, 85, 50, 85, 50, 85, 50, 85, 50, 89, 50, 85, 50, 85, 0, 15, 45, 60, 65, 70, 75, 80, 0, 15, 45, 60, 65, 70, 75, 80, 0, 15, 45, 60, 65, 70, 75, 80, 0, 15, 45, 60, 65, 70, 75, 80, 0, 56]) ],
           'M':[[*a,0] for a in zip( [340, 345, 350, 360, 525, 530, 535, 604, 614], [12, 18, 48, 48, 48, 30, 28, 28, 28] , [110, 110, 110, 120, 120, 120, 120, 12, 18] , [53, 46, 38, 97, 53, 43, 33, 148, 116] ) ]
}


cept_systems = {'Mars E-1 Config 1' : [	[	535,	28,	120,	33,	0. ] , # 29988 satellites
			[	530,	28,	120,	43,		0 ] ,
			[	525,	28,	120,	53,		0 ] ,
			[	360,	30,	120,	96.9,	0 ] ,
			[	350,	48,	110,	38,		0 ] ,
			[	345,	48,	110,	46,		0 ] ,
			[	340,	48,	110,	53,		0 ] ,
			[	604,	12,	12,	148,		0 ] ,
			[	614,	18,	18,	115.7,		0 ] ] }

if __name__ == '__main__' :
    # Example one : To write TLE definitions, using default paramaters and a selection. 
	# Note that the required parameters systems_information and system_names are set to
	# defaults systems_5Cs142dE_20241108 and recommended_system_names but can be any 
	# viable dictionaries. 
	# Issue the below commands to generate a default tle file:
    selection		= ['A','B','D']
    tle_df = create_tle_from_system_selection( selection , output_file = "constellation_systems-" + '-'.join(selection) + ".tle" )

	# example two : Here the functionallity is detailed in greater depth
    selection = ['A','B','I'] # ( A and M are variations of the same system )
    study_systems = [ systems_5Cs142dE_20241108[sys] for sys in selection ]
    print ( f'Will attempt to study systems {", ".join(selection)} corresponding to {", ".join([recommended_system_names[s] for s in selection])} respectively' )
    print ( "With the following data" , study_systems )
    print( repack_input(selection,study_systems) )

    # Example constellation definitions (Systems A–C)
    example_input = [
        # system, height_km, n_planes, sats_per_plane, inclination_deg, raan0_deg
        ["A", 525, 28, 120, 53.0, 0.0],
        ["B", 610, 36, 36, 42.0, 0.0],
        ["RT", 1200, 36, 40, 88.0, 0.0],
    ]
    constellation_df = build_constellation_df( example_input )
    print(constellation_df)

    constellation_df = build_constellation_df( repack_input( selection , study_systems ) )
    print(constellation_df)

    print("Generating TLEs...")
    tle_df = generate_constellation_tles(
        constellation_df,
        satnum_start=20000
    )

    print(f"Generated {len(tle_df)} satellites\n")

    # Show first few TLEs
    for _, row in tle_df.head(3).iterrows():
        print(row.tle1)
        print(row.tle2)
        print()

    # Optional: write to file
    output_file = "constellation.tle"
    with open(output_file, "w") as f:
        for _, row in tle_df.iterrows():
            f.write(row.tle1 + "\n")
            f.write(row.tle2 + "\n")

    print(f"TLEs written to {output_file}")
