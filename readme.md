<div id="top"></div>
<br />
<div align="center">
  <a href="https://github.com/leoxiaoyuan/ANN_for_welding">
    <img src="image/Logo.jpg" alt="Logo" width="320" height="160">
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
This project aimed to develop a surrogate model to predict residual stress of a single weld bead on a plate using the welding voltage, current, travel speed, bead length and thermal efficiency of welding.


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

### Repository structure
<img align="center" src="image/Structure.jpg" >
<br/>
<br/>
Folder <strong>extracted_data: </strong><br/>
contains 150 sets of data of residual stress of sampled welding simulations. with_heat_input means only the welding length, welding speed and welding heat input are deemed as input variables. with_V_C_E means the heat input is devided into there other variables, which are welding current, welding voltage and welding thermal efficiency. So the input variables of this dataset are welding lenth, travel speed, welding current, welding voltage and welding thermal efficiency.
<br/>
<br/>
Folder <strong>data_analysis: </strong><br/>
contains scripts for predicting residual stress.


### Usage

Run residual_stress_prediction.py 

<p align="right">(<a href="#top">back to top</a>)</p>










