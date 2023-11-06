@echo off
SETLOCAL EnableDelayedExpansion
set LOGFILE=batch.log
call :LOG > %LOGFILE%
exit /B

:LOG

python apriori.py -f ..\dataset\datasetA.data -t 1 -s 0.002
python apriori.py -f ..\dataset\datasetA.data -t 2 -s 0.002
python apriori.py -f ..\dataset\datasetA.data -t 1 -s 0.005
python apriori.py -f ..\dataset\datasetA.data -t 2 -s 0.005
python apriori.py -f ..\dataset\datasetA.data -t 1 -s 0.01
python apriori.py -f ..\dataset\datasetA.data -t 2 -s 0.01
python apriori.py -f ..\dataset\datasetB.data -t 1 -s 0.0015
python apriori.py -f ..\dataset\datasetB.data -t 2 -s 0.0015
python apriori.py -f ..\dataset\datasetB.data -t 1 -s 0.002
python apriori.py -f ..\dataset\datasetB.data -t 2 -s 0.002
python apriori.py -f ..\dataset\datasetB.data -t 1 -s 0.005
python apriori.py -f ..\dataset\datasetB.data -t 2 -s 0.005
python apriori.py -f ..\dataset\datasetC.data -t 1 -s 0.01
python apriori.py -f ..\dataset\datasetC.data -t 2 -s 0.01
python apriori.py -f ..\dataset\datasetC.data -t 1 -s 0.02
python apriori.py -f ..\dataset\datasetC.data -t 2 -s 0.02
python apriori.py -f ..\dataset\datasetC.data -t 1 -s 0.03
python apriori.py -f ..\dataset\datasetC.data -t 2 -s 0.03

pause