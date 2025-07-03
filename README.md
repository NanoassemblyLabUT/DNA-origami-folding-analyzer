# Cooperativity Calculator
A way to estimate cooperativity in DNA origami folding. Please reference our publication for the calculation.

## How to
1. run a real-time fluorometry experiment (qPCR) on your origami structure as well as the accompanying staples only
2. organize your data in same file format as the excel file "qPCR data for cooperativity" (download from repository)

     note: using this formatting, you can create multiple sheets within the same Excel file and the script will prompt you to select the desired sheet

4. run pyOrigamiBreak (1) (https://github.com/douglaslab/pyOrigamiBreak/tree/main) to get the theoretical Tfold output file
5. download cooperativity.py
6. run cooperativity.py and follow the accompanying prompts. A graph with Tfold steps will output accompanied by the cooperativity of the associated structure.
   
   note: due to noise, the script will sometimes miscalculate the first Tfold value (T1). There is an option to override this value (the code will prompt whether or not you would like to override T1) in order to more accurately represent what the cooperativity for the given structure is. 

## Citations
(1) Design principles for accurate folding of DNA origami
Aksel et al. (2024)
https://doi.org/10.1073/pnas.2406769121

## License
This version of cooperativity.py is available under the MIT license.
