@echo off
SETLOCAL EnableDelayedExpansion
set LOGFILE=batch.log
call :LOG > %LOGFILE%
exit /B

:LOG

python myEclat.py -f .\dataset\datasetA.data -s 0.002
python myEclat.py -f .\dataset\datasetA.data -s 0.002
python myEclat.py -f .\dataset\datasetA.data -s 0.005
python myEclat.py -f .\dataset\datasetA.data -s 0.005
python myEclat.py -f .\dataset\datasetA.data -s 0.01
python myEclat.py -f .\dataset\datasetA.data -s 0.01
python myEclat.py -f .\dataset\datasetB.data -s 0.0015
python myEclat.py -f .\dataset\datasetB.data -s 0.0015
python myEclat.py -f .\dataset\datasetB.data -s 0.002
python myEclat.py -f .\dataset\datasetB.data -s 0.002
python myEclat.py -f .\dataset\datasetB.data -s 0.005
python myEclat.py -f .\dataset\datasetB.data -s 0.005
python myEclat.py -f .\dataset\datasetC.data -s 0.01
python myEclat.py -f .\dataset\datasetC.data -s 0.01
python myEclat.py -f .\dataset\datasetC.data -s 0.02
python myEclat.py -f .\dataset\datasetC.data -s 0.02
python myEclat.py -f .\dataset\datasetC.data -s 0.03
python myEclat.py -f .\dataset\datasetC.data -s 0.03
python myEclat.py -f ..\dataset\datasetC.data -s 0.001

pause