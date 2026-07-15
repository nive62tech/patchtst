# Formula Documentation

## Mean Wave Period - Tm01 (Formula 1)

T_m01 = m0 / m1

Where m0 and m1 are the zeroth and first moments of the wave energy spectrum.
This is the ONLY wave period formula used in this project - it is the source
variable for both the forecasting target and the mwp_class classification label.

Cited from: "Wave measurements from ship mounted sensors in the Arctic marginal
ice zone"; "Utility of ocean wave parameters in ambient noise prediction"

**Peak Wave Period (Tp) is explicitly NOT used.** Tp = 1/fp = 1/f(max(E(f))) was
reviewed (cited from "Wave Measurements using Open Source Ship Mounted Ultrasonic
Altimeter - One Ocean Expedition" and "A Comprehensive Review of Phase-Averaged
and Phase-Resolving Wave Models for Coastal Modeling Applications") but
intentionally excluded from this project.

## Mean Wave Direction - theta_m (Formula 2)

theta_m = arctan( integral sin(theta) * G(f,theta) df dtheta / integral cos(theta) * G(f,theta) df dtheta )

This is the ECMWF/ERA5-standard definition, making it directly compatible with
the ERA5 dataset used in this project.

Cited from: "Wave Climate from Spectra and its Connections with Local and Remote
Wind Climate"; "Approximate Stokes Drift Profiles in Deep Water" (explicitly
notes this is the ECMWF standard)

## Circular Direction Encoding - sin/cos (Formula 3)

Raw direction in degrees is circular (359 deg and 1 deg are nearly the same
direction but numerically far apart), so direction variables are encoded as:

sin_X = sin(theta * pi/180)
cos_X = cos(theta * pi/180)

Applied to ALL THREE direction variables in the dataset:
- mwd (mean wave direction) -> sin_mwd, cos_mwd
- mdts (mean direction of total swell) -> sin_mdts, cos_mdts
- mdww (mean direction of wind waves) -> sin_mdww, cos_mdww

This gives 6 direction-derived channels total. Not paper-specific - this is
standard ML practice for circular variables, justified by the circular nature
confirmed in Formula 2.

Note: mdts and mdww are input features only - they are NOT forecasting
targets. The forecasting head predicts only swh, mwp, and raw mwd.
