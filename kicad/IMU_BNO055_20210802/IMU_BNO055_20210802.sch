EESchema Schematic File Version 4
EELAYER 30 0
EELAYER END
$Descr User 12176 8646
encoding utf-8
Sheet 1 1
Title "IMU BNO055"
Date "2021-07-31"
Rev "Ver.1.0"
Comp "N.T"
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L MCU_Microchip_ATmega:ATmega328P-AU U1
U 1 1 5F5DD62D
P 2700 3600
F 0 "U1" H 2700 2011 50  0000 C CNN
F 1 "ATmega328P-AU" H 2700 1920 50  0000 C CNN
F 2 "Package_QFP:TQFP-32_7x7mm_P0.8mm" H 2700 3600 50  0001 C CIN
F 3 "http://ww1.microchip.com/downloads/en/DeviceDoc/ATmega328_P%20AVR%20MCU%20with%20picoPower%20Technology%20Data%20Sheet%2040001984A.pdf" H 2700 3600 50  0001 C CNN
	1    2700 3600
	1    0    0    -1  
$EndComp
Wire Wire Line
	3300 4200 5100 4200
$Comp
L power:GND #PWR012
U 1 1 5F5ED2AC
P 2700 5400
F 0 "#PWR012" H 2700 5150 50  0001 C CNN
F 1 "GND" H 2705 5227 50  0000 C CNN
F 2 "" H 2700 5400 50  0001 C CNN
F 3 "" H 2700 5400 50  0001 C CNN
	1    2700 5400
	1    0    0    -1  
$EndComp
Wire Wire Line
	2700 5100 2700 5400
$Comp
L Connector_Generic:Conn_01x06 J1
U 1 1 5F5F60A3
P 3950 1550
F 0 "J1" V 3914 1162 50  0000 R CNN
F 1 "ICSP" V 3823 1162 50  0000 R CNN
F 2 "0.main.robot:JSTB6B-ZR" H 3950 1550 50  0001 C CNN
F 3 "~" H 3950 1550 50  0001 C CNN
	1    3950 1550
	0    -1   -1   0   
$EndComp
Wire Wire Line
	3750 1750 3750 2800
Wire Wire Line
	3750 2800 3300 2800
Wire Wire Line
	3300 2900 3950 2900
Wire Wire Line
	3950 2900 3950 1750
Wire Wire Line
	3300 2700 4050 2700
Wire Wire Line
	4050 2700 4050 1750
Wire Wire Line
	3300 3900 4150 3900
Wire Wire Line
	4150 3900 4150 1750
$Comp
L power:GND #PWR04
U 1 1 5F5F9536
P 4250 2000
F 0 "#PWR04" H 4250 1750 50  0001 C CNN
F 1 "GND" H 4255 1827 50  0000 C CNN
F 2 "" H 4250 2000 50  0001 C CNN
F 3 "" H 4250 2000 50  0001 C CNN
	1    4250 2000
	1    0    0    -1  
$EndComp
Wire Wire Line
	3850 1850 3850 1750
Wire Wire Line
	5000 1850 5000 1750
Wire Wire Line
	4250 1750 4250 1950
Wire Wire Line
	4250 1950 4900 1950
Connection ~ 4250 1950
Wire Wire Line
	4250 1950 4250 2000
$Comp
L Device:Resonator Y1
U 1 1 5F61753E
P 3700 3150
F 0 "Y1" V 3746 3261 50  0000 L CNN
F 1 "X`Tal 16MHz" V 3400 2850 50  0000 L CNN
F 2 "Crystal:Resonator_SMD_muRata_CSTxExxV-3Pin_3.0x1.1mm_HandSoldering" H 3675 3150 50  0001 C CNN
F 3 "~" H 3675 3150 50  0001 C CNN
	1    3700 3150
	0    -1   -1   0   
$EndComp
Wire Wire Line
	3300 3000 3700 3000
Wire Wire Line
	3300 3100 3500 3100
Wire Wire Line
	3500 3100 3500 3300
Wire Wire Line
	3500 3300 3700 3300
$Comp
L power:GND #PWR09
U 1 1 5F61F525
P 4000 3200
F 0 "#PWR09" H 4000 2950 50  0001 C CNN
F 1 "GND" H 4005 3027 50  0000 C CNN
F 2 "" H 4000 3200 50  0001 C CNN
F 3 "" H 4000 3200 50  0001 C CNN
	1    4000 3200
	1    0    0    -1  
$EndComp
Wire Wire Line
	3900 3150 4000 3150
Wire Wire Line
	4000 3150 4000 3200
Wire Wire Line
	2700 1850 2700 2000
Wire Wire Line
	2700 2000 2800 2000
Wire Wire Line
	2800 2000 2800 2100
Wire Wire Line
	2700 2000 2700 2100
Connection ~ 2700 2000
Wire Wire Line
	3300 2400 5300 2400
Wire Wire Line
	5300 2400 5300 1750
$Comp
L Device:C C1
U 1 1 5F632813
P 1800 2150
F 0 "C1" H 1915 2196 50  0000 L CNN
F 1 "0.1u" H 1915 2105 50  0000 L CNN
F 2 "Capacitor_SMD:C_0603_1608Metric_Pad1.05x0.95mm_HandSolder" H 1838 2000 50  0001 C CNN
F 3 "~" H 1800 2150 50  0001 C CNN
	1    1800 2150
	1    0    0    -1  
$EndComp
Wire Wire Line
	2700 2000 1800 2000
$Comp
L power:GND #PWR05
U 1 1 5F635929
P 1800 2450
F 0 "#PWR05" H 1800 2200 50  0001 C CNN
F 1 "GND" H 1805 2277 50  0000 C CNN
F 2 "" H 1800 2450 50  0001 C CNN
F 3 "" H 1800 2450 50  0001 C CNN
	1    1800 2450
	1    0    0    -1  
$EndComp
Wire Wire Line
	1800 2300 1800 2450
NoConn ~ 3300 3300
NoConn ~ 3300 3400
NoConn ~ 3300 3500
NoConn ~ 3300 3600
NoConn ~ 3300 4300
NoConn ~ 2100 2400
NoConn ~ 2100 2600
NoConn ~ 2100 2700
$Comp
L power:PWR_FLAG #FLG01
U 1 1 5F679026
P 8850 2450
F 0 "#FLG01" H 8850 2525 50  0001 C CNN
F 1 "PWR_FLAG" H 8850 2623 50  0000 C CNN
F 2 "" H 8850 2450 50  0001 C CNN
F 3 "~" H 8850 2450 50  0001 C CNN
	1    8850 2450
	1    0    0    -1  
$EndComp
$Comp
L power:PWR_FLAG #FLG02
U 1 1 5F6792D1
P 9250 2550
F 0 "#FLG02" H 9250 2625 50  0001 C CNN
F 1 "PWR_FLAG" H 9250 2723 50  0000 C CNN
F 2 "" H 9250 2550 50  0001 C CNN
F 3 "~" H 9250 2550 50  0001 C CNN
	1    9250 2550
	-1   0    0    1   
$EndComp
$Comp
L power:+5V #PWR06
U 1 1 5F67DE0D
P 9250 2450
F 0 "#PWR06" H 9250 2300 50  0001 C CNN
F 1 "+5V" H 9265 2623 50  0000 C CNN
F 2 "" H 9250 2450 50  0001 C CNN
F 3 "" H 9250 2450 50  0001 C CNN
	1    9250 2450
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR07
U 1 1 5F67E36B
P 8850 2550
F 0 "#PWR07" H 8850 2300 50  0001 C CNN
F 1 "GND" H 8855 2377 50  0000 C CNN
F 2 "" H 8850 2550 50  0001 C CNN
F 3 "" H 8850 2550 50  0001 C CNN
	1    8850 2550
	1    0    0    -1  
$EndComp
Wire Wire Line
	8850 2450 8850 2550
Wire Wire Line
	9250 2450 9250 2550
Text Notes 3450 4050 0    50   ~ 0
(RX)
Text Notes 3450 4300 0    50   ~ 0
(TX)
$Comp
L Device:LED D2
U 1 1 5F5BED46
P 3550 5300
F 0 "D2" V 3589 5183 50  0000 R CNN
F 1 "Active" V 3498 5183 50  0000 R CNN
F 2 "LED_SMD:LED_0603_1608Metric_Pad1.05x0.95mm_HandSolder" H 3550 5300 50  0001 C CNN
F 3 "~" H 3550 5300 50  0001 C CNN
	1    3550 5300
	0    -1   -1   0   
$EndComp
$Comp
L Device:R R1
U 1 1 5F5BF7D9
P 3550 4950
F 0 "R1" V 3343 4950 50  0000 C CNN
F 1 "470" V 3434 4950 50  0000 C CNN
F 2 "Resistor_SMD:R_0603_1608Metric_Pad1.05x0.95mm_HandSolder" V 3480 4950 50  0001 C CNN
F 3 "~" H 3550 4950 50  0001 C CNN
	1    3550 4950
	-1   0    0    1   
$EndComp
$Comp
L power:GND #PWR013
U 1 1 5F5BFFBA
P 3550 5600
F 0 "#PWR013" H 3550 5350 50  0001 C CNN
F 1 "GND" H 3555 5427 50  0000 C CNN
F 2 "" H 3550 5600 50  0001 C CNN
F 3 "" H 3550 5600 50  0001 C CNN
	1    3550 5600
	1    0    0    -1  
$EndComp
Wire Wire Line
	3300 4800 3550 4800
Wire Wire Line
	3550 5100 3550 5150
Wire Wire Line
	3550 5450 3550 5600
Text Notes 5200 2250 1    50   ~ 0
(RX)
Text Notes 5050 2250 1    50   ~ 0
(TX)
$Comp
L power:+5V #PWR02
U 1 1 5F5E382F
P 4650 1550
F 0 "#PWR02" H 4650 1400 50  0001 C CNN
F 1 "+5V" H 4665 1723 50  0000 C CNN
F 2 "" H 4650 1550 50  0001 C CNN
F 3 "" H 4650 1550 50  0001 C CNN
	1    4650 1550
	1    0    0    -1  
$EndComp
Wire Wire Line
	4650 1550 4650 1850
Wire Wire Line
	4650 1850 5000 1850
Wire Wire Line
	2700 1850 3850 1850
Wire Wire Line
	2700 1650 2700 1850
Connection ~ 2700 1850
$Comp
L power:+5V #PWR01
U 1 1 61065FFE
P 3200 1550
F 0 "#PWR01" H 3200 1400 50  0001 C CNN
F 1 "+5V" H 3215 1723 50  0000 C CNN
F 2 "" H 3200 1550 50  0001 C CNN
F 3 "" H 3200 1550 50  0001 C CNN
	1    3200 1550
	1    0    0    -1  
$EndComp
Wire Wire Line
	3200 1550 3200 1650
Wire Wire Line
	5100 1750 5100 4200
Wire Wire Line
	3300 3700 5500 3700
Wire Wire Line
	5500 3700 5500 3900
Wire Wire Line
	5500 3900 6050 3900
Text Notes 6200 3750 0    50   ~ 0
(SCL)
Text Notes 6200 3900 0    50   ~ 0
(SDA)
$Comp
L power:+5V #PWR010
U 1 1 61093882
P 6600 3200
F 0 "#PWR010" H 6600 3050 50  0001 C CNN
F 1 "+5V" H 6615 3373 50  0000 C CNN
F 2 "" H 6600 3200 50  0001 C CNN
F 3 "" H 6600 3200 50  0001 C CNN
	1    6600 3200
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR011
U 1 1 61094070
P 6600 4500
F 0 "#PWR011" H 6600 4250 50  0001 C CNN
F 1 "GND" H 6605 4327 50  0000 C CNN
F 2 "" H 6600 4500 50  0001 C CNN
F 3 "" H 6600 4500 50  0001 C CNN
	1    6600 4500
	1    0    0    -1  
$EndComp
Wire Wire Line
	6600 3200 6600 3250
$Comp
L Device:R R5
U 1 1 61098D1D
P 5750 3500
F 0 "R5" H 5820 3546 50  0000 L CNN
F 1 "47k" H 5820 3455 50  0000 L CNN
F 2 "Resistor_SMD:R_0603_1608Metric_Pad1.05x0.95mm_HandSolder" V 5680 3500 50  0001 C CNN
F 3 "~" H 5750 3500 50  0001 C CNN
	1    5750 3500
	1    0    0    -1  
$EndComp
$Comp
L Device:R R6
U 1 1 6109925F
P 6050 3500
F 0 "R6" H 6120 3546 50  0000 L CNN
F 1 "47k" H 6120 3455 50  0000 L CNN
F 2 "Resistor_SMD:R_0603_1608Metric_Pad1.05x0.95mm_HandSolder" V 5980 3500 50  0001 C CNN
F 3 "~" H 6050 3500 50  0001 C CNN
	1    6050 3500
	1    0    0    -1  
$EndComp
Wire Wire Line
	5750 3350 5750 3250
Wire Wire Line
	5750 3250 6050 3250
Connection ~ 6600 3250
Wire Wire Line
	6600 3250 6600 3600
Wire Wire Line
	6050 3250 6050 3350
Connection ~ 6050 3250
Wire Wire Line
	6050 3250 6600 3250
Wire Wire Line
	5750 3650 5750 3800
Connection ~ 5750 3800
Wire Wire Line
	6050 3650 6050 3900
Connection ~ 6050 3900
Wire Wire Line
	3300 3800 5750 3800
Wire Wire Line
	4900 1750 4900 1950
Wire Wire Line
	3300 4100 5200 4100
Wire Wire Line
	5200 1750 5200 4100
$Comp
L Connector_Generic:Conn_01x05 J2
U 1 1 610A8AAA
P 5100 1550
F 0 "J2" V 5064 1262 50  0000 R CNN
F 1 "SerialOUT" V 5100 2250 50  0000 R CNN
F 2 "Connector_JST:JST_PH_S5B-PH-K_1x05_P2.00mm_Horizontal" H 5100 1550 50  0001 C CNN
F 3 "~" H 5100 1550 50  0001 C CNN
	1    5100 1550
	0    1    -1   0   
$EndComp
$Comp
L Connector_Generic:Conn_01x08 J3
U 1 1 610AA62D
P 7400 3900
F 0 "J3" H 7480 3892 50  0000 L CNN
F 1 "GY-BNO055" H 7480 3801 50  0000 L CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x08_P2.54mm_Vertical" H 7400 3900 50  0001 C CNN
F 3 "~" H 7400 3900 50  0001 C CNN
	1    7400 3900
	1    0    0    -1  
$EndComp
Wire Wire Line
	6600 3600 7200 3600
Wire Wire Line
	5750 3800 7200 3800
Wire Wire Line
	6050 3900 7200 3900
Wire Wire Line
	7200 3700 6600 3700
Wire Wire Line
	6600 3700 6600 4000
$Comp
L Device:LED D3
U 1 1 610BF58B
P 4250 5300
F 0 "D3" V 4289 5183 50  0000 R CNN
F 1 "Left" V 4198 5183 50  0000 R CNN
F 2 "LED_SMD:LED_0603_1608Metric_Pad1.05x0.95mm_HandSolder" H 4250 5300 50  0001 C CNN
F 3 "~" H 4250 5300 50  0001 C CNN
	1    4250 5300
	0    -1   -1   0   
$EndComp
$Comp
L Device:R R2
U 1 1 610BF7C7
P 4250 4950
F 0 "R2" V 4043 4950 50  0000 C CNN
F 1 "470" V 4134 4950 50  0000 C CNN
F 2 "Resistor_SMD:R_0603_1608Metric_Pad1.05x0.95mm_HandSolder" V 4180 4950 50  0001 C CNN
F 3 "~" H 4250 4950 50  0001 C CNN
	1    4250 4950
	-1   0    0    1   
$EndComp
$Comp
L power:GND #PWR014
U 1 1 610BF7D1
P 4250 5600
F 0 "#PWR014" H 4250 5350 50  0001 C CNN
F 1 "GND" H 4255 5427 50  0000 C CNN
F 2 "" H 4250 5600 50  0001 C CNN
F 3 "" H 4250 5600 50  0001 C CNN
	1    4250 5600
	1    0    0    -1  
$EndComp
Wire Wire Line
	4250 5100 4250 5150
Wire Wire Line
	4250 5450 4250 5600
$Comp
L Device:LED D4
U 1 1 610C1F0C
P 4750 5300
F 0 "D4" V 4789 5183 50  0000 R CNN
F 1 "Center" V 4698 5183 50  0000 R CNN
F 2 "LED_SMD:LED_0603_1608Metric_Pad1.05x0.95mm_HandSolder" H 4750 5300 50  0001 C CNN
F 3 "~" H 4750 5300 50  0001 C CNN
	1    4750 5300
	0    -1   -1   0   
$EndComp
$Comp
L Device:R R3
U 1 1 610C217C
P 4750 4950
F 0 "R3" V 4543 4950 50  0000 C CNN
F 1 "470" V 4634 4950 50  0000 C CNN
F 2 "Resistor_SMD:R_0603_1608Metric_Pad1.05x0.95mm_HandSolder" V 4680 4950 50  0001 C CNN
F 3 "~" H 4750 4950 50  0001 C CNN
	1    4750 4950
	-1   0    0    1   
$EndComp
$Comp
L power:GND #PWR015
U 1 1 610C2186
P 4750 5600
F 0 "#PWR015" H 4750 5350 50  0001 C CNN
F 1 "GND" H 4755 5427 50  0000 C CNN
F 2 "" H 4750 5600 50  0001 C CNN
F 3 "" H 4750 5600 50  0001 C CNN
	1    4750 5600
	1    0    0    -1  
$EndComp
Wire Wire Line
	4750 5100 4750 5150
Wire Wire Line
	4750 5450 4750 5600
$Comp
L Device:LED D5
U 1 1 610C3EFE
P 5250 5300
F 0 "D5" V 5289 5183 50  0000 R CNN
F 1 "Right" V 5198 5183 50  0000 R CNN
F 2 "LED_SMD:LED_0603_1608Metric_Pad1.05x0.95mm_HandSolder" H 5250 5300 50  0001 C CNN
F 3 "~" H 5250 5300 50  0001 C CNN
	1    5250 5300
	0    -1   -1   0   
$EndComp
$Comp
L Device:R R4
U 1 1 610C41A2
P 5250 4950
F 0 "R4" V 5043 4950 50  0000 C CNN
F 1 "470" V 5134 4950 50  0000 C CNN
F 2 "Resistor_SMD:R_0603_1608Metric_Pad1.05x0.95mm_HandSolder" V 5180 4950 50  0001 C CNN
F 3 "~" H 5250 4950 50  0001 C CNN
	1    5250 4950
	-1   0    0    1   
$EndComp
$Comp
L power:GND #PWR016
U 1 1 610C41AC
P 5250 5600
F 0 "#PWR016" H 5250 5350 50  0001 C CNN
F 1 "GND" H 5255 5427 50  0000 C CNN
F 2 "" H 5250 5600 50  0001 C CNN
F 3 "" H 5250 5600 50  0001 C CNN
	1    5250 5600
	1    0    0    -1  
$EndComp
Wire Wire Line
	5250 5100 5250 5150
Wire Wire Line
	5250 5450 5250 5600
Wire Wire Line
	3300 4700 4250 4700
Wire Wire Line
	4250 4700 4250 4800
Wire Wire Line
	3300 4600 4750 4600
Wire Wire Line
	4750 4600 4750 4800
Wire Wire Line
	3300 4500 5250 4500
Wire Wire Line
	5250 4500 5250 4800
Text Notes 3300 4900 0    50   ~ 0
(D7)
Text Notes 4000 4800 0    50   ~ 0
(D6)
Text Notes 4500 4700 0    50   ~ 0
(D5)
Text Notes 5000 4600 0    50   ~ 0
(D4)
Wire Wire Line
	3300 4400 7200 4400
Wire Wire Line
	7200 4400 7200 4300
Text Notes 5350 4350 0    50   ~ 0
(D3)
Text Notes 6750 4350 0    50   ~ 0
(Reset)
NoConn ~ 7200 4100
NoConn ~ 7200 4200
Wire Wire Line
	7200 4000 6600 4000
Connection ~ 6600 4000
Wire Wire Line
	6600 4000 6600 4500
Text Notes 6750 4100 0    50   ~ 0
(ADDR)
$Comp
L Switch:SW_Push SW1
U 1 1 610D763D
P 6700 2500
F 0 "SW1" H 6700 2785 50  0000 C CNN
F 1 "CAL" H 6700 2694 50  0000 C CNN
F 2 "0.main.robot:TVAF06-A020B-R" H 6700 2700 50  0001 C CNN
F 3 "~" H 6700 2700 50  0001 C CNN
	1    6700 2500
	1    0    0    -1  
$EndComp
$Comp
L Switch:SW_Push SW2
U 1 1 610D9E53
P 7400 2600
F 0 "SW2" H 7400 2885 50  0000 C CNN
F 1 "SET" H 7400 2794 50  0000 C CNN
F 2 "0.main.robot:TVAF06-A020B-R" H 7400 2800 50  0001 C CNN
F 3 "~" H 7400 2800 50  0001 C CNN
	1    7400 2600
	1    0    0    -1  
$EndComp
Wire Wire Line
	3300 2500 5750 2500
Wire Wire Line
	3300 2600 6100 2600
$Comp
L Device:R R7
U 1 1 610E01FF
P 5750 2300
F 0 "R7" H 5820 2346 50  0000 L CNN
F 1 "47k" H 5820 2255 50  0000 L CNN
F 2 "Resistor_SMD:R_0603_1608Metric_Pad1.05x0.95mm_HandSolder" V 5680 2300 50  0001 C CNN
F 3 "~" H 5750 2300 50  0001 C CNN
	1    5750 2300
	1    0    0    -1  
$EndComp
$Comp
L Device:R R8
U 1 1 610E0928
P 6100 2300
F 0 "R8" H 6170 2346 50  0000 L CNN
F 1 "47k" H 6170 2255 50  0000 L CNN
F 2 "Resistor_SMD:R_0603_1608Metric_Pad1.05x0.95mm_HandSolder" V 6030 2300 50  0001 C CNN
F 3 "~" H 6100 2300 50  0001 C CNN
	1    6100 2300
	1    0    0    -1  
$EndComp
Wire Wire Line
	5750 2450 5750 2500
Connection ~ 5750 2500
Wire Wire Line
	5750 2500 6500 2500
Wire Wire Line
	6100 2450 6100 2600
Connection ~ 6100 2600
Wire Wire Line
	6100 2600 7200 2600
$Comp
L power:+5V #PWR03
U 1 1 610E47D9
P 6100 1950
F 0 "#PWR03" H 6100 1800 50  0001 C CNN
F 1 "+5V" H 6115 2123 50  0000 C CNN
F 2 "" H 6100 1950 50  0001 C CNN
F 3 "" H 6100 1950 50  0001 C CNN
	1    6100 1950
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR08
U 1 1 610E4C0F
P 7700 2800
F 0 "#PWR08" H 7700 2550 50  0001 C CNN
F 1 "GND" H 7705 2627 50  0000 C CNN
F 2 "" H 7700 2800 50  0001 C CNN
F 3 "" H 7700 2800 50  0001 C CNN
	1    7700 2800
	1    0    0    -1  
$EndComp
Wire Wire Line
	6100 1950 6100 2050
Wire Wire Line
	6100 2050 5750 2050
Wire Wire Line
	5750 2050 5750 2150
Connection ~ 6100 2050
Wire Wire Line
	6100 2050 6100 2150
Wire Wire Line
	7600 2600 7700 2600
Wire Wire Line
	7700 2600 7700 2750
Wire Wire Line
	6900 2500 7050 2500
Wire Wire Line
	7050 2500 7050 2750
Wire Wire Line
	7050 2750 7700 2750
Connection ~ 7700 2750
Wire Wire Line
	7700 2750 7700 2800
Wire Wire Line
	2700 1650 3200 1650
$EndSCHEMATC