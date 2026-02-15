# Sources (for criteria + evidence-based prep)

This project references official criteria pages/PDFs and peer-reviewed literature. The app only stores short summaries and uses conservative recommendations.

- WNBF UK Men's Physique criteria (PDF): https://wnbfuk.com/wp-content/uploads/2021/05/WNBF-UK-MENS-PHYSIQUE-CRITERIA-1.pdf
- PCA criteria page: https://www.pcaofficial.com/criteria

Evidence-based contest prep / peak week:
- Helms et al. 2014 (JISSN): https://pmc.ncbi.nlm.nih.gov/articles/PMC4033492/
- Escalante et al. 2021 peak week review: https://pmc.ncbi.nlm.nih.gov/articles/PMC8201693/
- Chappell et al. 2018 peak week strategies (survey): https://shura.shu.ac.uk/23099/1/sports-06-00126-v2.pdf

# Kinect v2 (Linux) notes

Hardware requirements / stability tips:
- Use a direct USB 3.0 port (no hubs/docks); Kinect v2 is very bandwidth sensitive.
- Ensure the Kinect power supply is original or high-quality.
- Avoid long/cheap USB3 cables; I/O errors often point to cable/port issues.
- If you see `LIBUSB_ERROR_IO`, re-seat the cable and change ports.
- Prefer OpenGL/OpenCL packet pipelines for FPS; CPU is a fallback.
- Consider increasing usbfs memory (example): `options usbcore usbfs_memory_mb=1024`.
