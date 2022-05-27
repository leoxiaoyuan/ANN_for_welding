<div id="top"></div>
<br />
<div align="center">
  <a href="https://github.com/leoxiaoyuan/ANN_for_welding">
    <img src="https://github.com/leoxiaoyuan/ANN_for_welding/blob/main/image/Logo.jpg">
  </a>

  <h3 align="center">ANN for welding</h3>

  <p align="center">
    An awesome sorregate model to predict welding residual stress
    <br />
    <a href="https://github.com/leoxiaoyuan/ANN_for_welding"><strong>Explore the model »</strong></a>
    <br />
    <br />
  </p>
</div>




### About The Project
This project aimed to develop a surrogate model to predict residual stress of a single weld bead on a plate using arc travel length, travel speed and net power input rate.


### Prerequisites

Tensorflow and Keras are required to be installed
* pip3
  ```sh
  pip3 install tensorflow
  ```

### Installation

 Clone the repo to your local directory
   ```sh
   git clone https://github.com/leoxiaoyuan/ANN_for_welding.git YOUR_DIRECTORY
   ```

### Introduction to main folders
Folder <strong>Abaqus_scripts: </strong><br/>
contains scripts that used for creating thermal and mechanical analysis input files and collecting longitudinal stresses on B-D line. Subroutine used for defining the heat source model is stored in Thermal_inp_files folder. The shell scripts for submitting the thermal and mechanical job arrays to CSF are stored in the relevant folders.
<br/>
<br/>
Folder <strong>ANN_results: </strong><br/>
contains the final tuned ANN model and predictions on the test dataset.
<br/>
<br/>
Folder <strong>extracted_data: </strong><br/>
contains 205 sets of data of sampled welding simulations.
<br/>
<br/>
Folder <strong>data_analysis: </strong><br/>
contains scripts for predicting residual stress.
<br/>
<br/>
Folder <strong>Surrogate_model_modeling: </strong><br/>
contains two scripts which built a ANN and a Gaussian Process model.

### Usage

:triangular_flag_on_post:The use of the developed ANN surrogate model <br/>
&emsp;:one: Open 'ANN_results\saved_model\Predict.py'<br/>
&emsp;:two: Change the welding parameters to those to be predicted<br/>
&emsp;:three: Run the python script, and the predicted stresses will be printed out<br/>
<br/>
:triangular_flag_on_post: The use of Abaqus python scripts to generate simulations <br/>
&emsp;:one: Open Abaqus command terminal <br/>
&emsp;:two: Change the directory to '\~/Abaqus_scripts/' <br/>
&emsp;:three: Run the command 'Abaqus cae noGUI=Create_input_files.py -- a b c d e f'<br/>
&emsp;&emsp;a, b, c, d, e, f refer to specimen length(m), width(m), height(m), welding length(m), welding speed(m/s), net energy input rate(W) respectively<br/>
&emsp;:four: Submit thermal analysis simulation by running Run_Thermal <br/>
&emsp;:five: Submit mechanical analysis simulation by running Run_Mechanical <br/>
<br/>
:triangular_flag_on_post:The use of Abaqus python scripts to collecting data form simulations<br/>
&emsp;:one: Open Abaqus command terminal <br/>
&emsp;:two: Change the directory to '\~/Abaqus_scripts/Mechanical_inp_files/' <br/>
&emsp;:three: Run the command 'Abaqus cae script=extract_data.py'<br/>
<p align="right">(<a href="#top">back to top</a>)</p>










