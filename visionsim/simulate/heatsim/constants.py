# Vendored from heat-sim-blender:addon/lib/constants.py @ e5b4afe
import math

THERMAL_VIEW_LAYER_NAME = "HeatSim Thermal"
THERMAL_MATERIAL_NAME = "HeatSim_Thermal_Override"
IRRADIANCE_LAYER_NAME = "HeatSim_Irradiance"
ALBEDO_LAYER_NAME = "HeatSim_Albedo"
BAKE_UV_LAYER_NAME = "HeatSim_Bake_UV"
ATLAS_UV_LAYER_NAME = "HeatSim_Atlas_UV"
# The render-time atlas image (adapter.write_atlas writes the EXR; blender.py loads +
# packs it under this datablock name; thermal_shader.py's temperature-source chain
# samples it by this name).
ATLAS_IMAGE_NAME = "HeatSim_Temperature_Atlas"
# OBJECT-domain custom property (float 0.0/1.0): explicitly gates the shader's atlas
# mix factor to 0 for any object that is not an atlas participant, independent of
# whatever the atlas image happens to contain at the (0,0,0) UV a missing
# HeatSim_Atlas_UV attribute defaults to. Stamped on every mesh by
# adapter.write_frame_attributes whenever an atlas_plan is supplied.
ATLAS_COVERAGE_PROP = "heatsim_atlas_coverage"


CAMERA = 'Boson'
HFOV = 24.3
VFOV = 19.5

SIGMA = 5.6704e-8/(1000**2) # W/(mm^2*K^4)
AMBIENT_TEMP = 295.372

# Cycles' DIFFUSE/DIRECT+INDIRECT bake stores outgoing radiance L_out
# for an implicit ρ=1 Lambertian receiver, in W/m²/sr. The thermal heat
# equation wants irradiance E in W/m². The Lambert hemisphere integral
# gives E = π · L_out, so we multiply baked irradiance pixels by π at
# ingest to recover physical irradiance. The Direct Kernel produces
# E directly (no factor needed). NOT applied to the COLOR/albedo bake —
# that pass stores ρ ∈ [0, 1] which is dimensionless.
CYCLES_LOUT_TO_IRRADIANCE = math.pi

# Just took this value from google for a room at 25c. The typical value ranges from 10 to 100
CONVECTION_COEFF = 0/(1000**2) # W/(mm^2*K)


# Density of the material in kg/mm^3 at 98 celcius
Density = {
    "aluminium": 2700/(1000**3),
    "pvc": 1330/(1000**3),
    "glass": 2500/(1000**3),
    "copper": 8960/(1000**3),
    "polystyrene": 1050/(1000**3),
    "wood": 897/(1000**3),
    "steel": 7930/(1000**3),
    'brick': 2200/(1000**3),
    "concrete": 2300/(1000**3),
    "plaster": 1200/(1000**3),
    "asphalt": 2100/(1000**3),
    "iron": 7200/(1000**3),
    "li_ion": 2500/(1000**3),
}
# Specific heat of the material in J/(kg*K) at 98 celcius
SpecificHeat = {
    "aluminium": 978,
    "pvc": 880,
    "glass": 840,
    "copper": 385,
    "polystyrene": 1300,
    "wood": 2380,
    "steel": 280,
    "brick": 800,
    "concrete": 880,
    "plaster": 1090,
    "asphalt": 920,
    "iron": 450,
    "li_ion": 1100,
}
Absorptivity = {
    "aluminium": 1.0,
    "pvc": 1.0,
    "glass": 0.9,
    "copper": 0.9,
    "polystyrene": 0.9,
    "wood": 0.9,
    "steel": 0.9,
    "brick": 0.9,
    "concrete": 0.94,
    "plaster": 0.93,
    "asphalt": 0.95,
    "iron": 0.30,
    "li_ion": 0.88,
}

# Thermal diffusivity in mm^2/s (alpha = k / (rho * c))
TDiff = {
    "aluminium": 97.0, #97, #1.0, #97,
    "aluminium-6061": 64,
    "glass" : 0.34,
    "pvc": 0.17, # conductivity - 0.2 W/mK,
    "polystyrene": 0.5, #0.5,
    "copper": 111,
    "wood": 0.082,
    "steel": 4.2,
    "brick": 0.52,
    "concrete": 0.5,
    "plaster": 0.4,
    "asphalt": 0.3,
    "iron": 18.0,
    "li_ion": 0.2,
}


TConductivity = {
    "aluminium": TDiff["aluminium"]*(Density["aluminium"]*SpecificHeat["aluminium"]),
    "pvc": TDiff["pvc"]*(Density["pvc"]*SpecificHeat["pvc"]),
    "glass": TDiff["glass"]*(Density["glass"]*SpecificHeat["glass"]),
    "copper": TDiff["copper"]*(Density["copper"]*SpecificHeat["copper"]),
    "polystyrene": TDiff["polystyrene"]*(Density["polystyrene"]*SpecificHeat["polystyrene"]),
    "wood": TDiff["wood"]*(Density["wood"]*SpecificHeat["wood"]),
    "steel": TDiff["steel"]*(Density["steel"]*SpecificHeat["steel"]),
    "brick": TDiff["brick"]*(Density["brick"]*SpecificHeat["brick"]),
    "concrete": TDiff["concrete"]*(Density["concrete"]*SpecificHeat["concrete"]),
    "plaster": TDiff["plaster"]*(Density["plaster"]*SpecificHeat["plaster"]),
    "asphalt": TDiff["asphalt"]*(Density["asphalt"]*SpecificHeat["asphalt"]),
    "iron": TDiff["iron"]*(Density["iron"]*SpecificHeat["iron"]),
    "li_ion": TDiff["li_ion"]*(Density["li_ion"]*SpecificHeat["li_ion"]),
}
