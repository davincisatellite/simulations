$MODEL FinalModel
# 
# template file: Template.tpl
# ESATAN-TMS 2018 sp1, run date 17:44 Mon 19 Dec 2022
# Model name: FinalModel        Analysis case: acase01
#
  $LOCALS
  # GENCODE LOCALS - DO NOT REMOVE 
#
  $NODES
  # GENCODE NODES - DO NOT REMOVE 
#
  $CONDUCTORS
  # GENCODE CONDUCTORS - DO NOT REMOVE 
#
  $CONSTANTS
  # GENCODE CONSTANTS - DO NOT REMOVE 
#
  $ARRAYS
  # GENCODE ARRAYS - DO NOT REMOVE 
#
  $EVENTS
  # GENCODE EVENTS - DO NOT REMOVE 
#
  $SUBROUTINES
  # GENCODE SUBROUTINES - DO NOT REMOVE 
C
  $INITIAL
  # GENCODE BOUNDARY_CONDS - DO NOT REMOVE 
  # GENCODE INITIAL - DO NOT REMOVE 
C
  $EXECUTION
C
C Steady State Solution
C
      RELXCA=0.01
      NLOOP=100
C
      CALL SOLVFM
C
C Transient Solution
C
      TIMEND=PERIOD
      DTIMEI=TIMEND/100.0
      OUTINT=TIMEND/10.0
C
      CALL SLCRNC


C
  $VARIABLES1
  # GENCODE VARIABLES1 - DO NOT REMOVE 
C
  $VARIABLES2
  # GENCODE VARIABLES2 - DO NOT REMOVE 
C
  $OUTPUTS
      CALL PRNDTB(' ', 'L, T, QS, QE, QA, QI, C', CURRENT)
C
      CALL DMPTMD(' ', 'NODES, CONDUCTORS', CURRENT, ' ')


  # GENCODE OUTPUTS - DO NOT REMOVE 
C
$ENDMODEL FinalModel