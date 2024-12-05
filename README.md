# Work in Progress for Version 2. 
- Version 1 uses a more rudimentary ray tracing method where the image plane is placed behind the back of the BH. 
- Version 2 changes the ray tracing method to have a celestial sphere covering the entire BH scene instead.
- Version 2 Incorporates Correct Relativistic aberration which requires a DOT product first before the aberration formula.
- Researching on analytical solution to null geodesics instead of relying on solve_ivp from scipy which is slow and not parellizable in GPUs.

