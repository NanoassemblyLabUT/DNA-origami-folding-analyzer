# Cooperativity Calculator
A way to estimate cooperativity in DNA origami folding.

## How to
1. run a real-time fluorometry experiment (qPCR) on your origami structure as well as the accompanying staples only
2. organize your data in same file format as the excel file "qPCR data for cooperativity" (download from repository)
     note: using this formatting, you can create multiple sheets within the same Excel file and the script will ask you for which sheet you would like to specifically reference
3. run pyOrigamiBreak (1) (https://colab.research.google.com/drive/1wRAO8LdY5XCeuZfsmvdUHlJWMBQmQBVa) to get the theoretical Tfold output file
4. download cooperativity.py
5. run cooperativity.py and follow the accompanying prompts. A graph with Tfold steps will output accompanied by the cooperativity of the associated structure.

## Citations
(1) Design principles for accurate folding of DNA origami
Aksel et al. (2024)
https://doi.org/10.1073/pnas.2406769121

## License
This version of cooperativity.py is available under the MIT license.
