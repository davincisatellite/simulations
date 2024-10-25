      USE gar_GlobalArrays_m,
     &   AI => gar_arrayInfoList,
     &   AN => gar_arrayNames,
     &   AP => gar_arrayInfoPointers,
     &   CA => gar_arrayValuesChar,
     &   IA => gar_arrayValuesInt,
     &   RA => gar_arrayValuesReal,
     &   CCN => gar_controlConstNames,
     &   INCORE => gar_dynamicCoreInt,
     &   DYCORE => gar_dynamicCoreReal,
     &   MD => gar_modelDirectory,
     &   SN => gar_modelNames,
     &   MST => gar_modelStatuses,
     &   ALNAM => gar_nodeAliasNames,
     &   ALVAL => gar_nodeAliasNumbers,
     &   FPCS => gar_pcsFluid,
     &   FPCSP => gar_pcsPtrsFluid,
     &   NPCS => gar_pcsThermal,
     &   NPCSP => gar_pcsPtrsThermal,
     &   PCS => gar_pcsThermalOld,
     &   PCSP => gar_pcsPtrsThermalOld,
     &   VCN => gar_userConstNames,
     &   PU => gar_userConstPointersChar,
     &   CU => gar_userConstValuesChar,
     &   IU => gar_userConstValuesInt,
     &   RU => gar_userConstValuesReal
      USE gar_GlobalArrays_m,
     &   C => gar_thermalNodeCapacitances,
     &   FXT => gar_thermalNodeCoordsX,
     &   FYT => gar_thermalNodeCoordsY,
     &   FZT => gar_thermalNodeCoordsZ,
     &   EPS => gar_thermalNodeEmissivities,
     &   QAI => gar_thermalNodeIncidentFluxesAlbedo,
     &   QEI => gar_thermalNodeIncidentFluxesPlanet,
     &   QSI => gar_thermalNodeIncidentFluxesSolar,
     &   NLAB => gar_thermalNodeLabels,
     &   NNUM => gar_thermalNodeNums,
     &   ALP => gar_thermalNodeAbsorptivities,
     &   NST => gar_thermalNodeStatuses,
     &   NST2 => gar_thermalNodeStatusesCopy,
     &   A => gar_thermalNodeSurfaceAreas,
     &   T => gar_thermalNodeTemps,
     &   QA => gar_thermalNodeHeatSourcesAlbedo,
     &   QE => gar_thermalNodeHeatSourcesPlanet,
     &   QI => gar_thermalNodeHeatSourcesInternal,
     &   QR => gar_thermalNodeHeatSourcesOther,
     &   QS => gar_thermalNodeHeatSourcesSolar,
     &   CUD => gar_thermalNodeUserEntityValsChar,
     &   IUD => gar_thermalNodeUserEntityValsInt,
     &   RUD => gar_thermalNodeUserEntityValsReal
      USE gar_GlobalArrays_m,
     &   CMP => gar_fluidNodeCompliances,
     &   FX => gar_fluidNodeCoordsX,
     &   FY => gar_fluidNodeCoordsY,
     &   FZ => gar_fluidNodeCoordsZ,
     &   FLA => gar_fluidNodeFlowAreas,
     &   FRG => gar_fluidNodeFlowRegimes,
     &   FQ => gar_fluidNodeHeatSources,
     &   FA => gar_fluidNodeHeatTransferAreas,
     &   FD => gar_fluidNodeHydraulicDiameters,
     &   L => gar_fluidNodeLabels,
     &   FL => gar_fluidNodeLengths,
     &   FM => gar_fluidNodeMassFlowSources,
     &   FH => gar_fluidNodeMassSourceSpecEnthalpies,
     &   FW => gar_fluidNodeMassSourceSpecHumidities,
     &   FR => gar_fluidNodeMassSourceTemps,
     &   MODNOD => gar_fluidNodeModelNums,
     &   FNNUM => gar_fluidNodeNums,
     &   P => gar_fluidNodePressures,
     &   PHI => gar_fluidNodeRelHumidities,
     &   FE => gar_fluidNodeSpecEnthalpies,
     &   SPHUM => gar_fluidNodeSpecHumidities,
     &   FST => gar_fluidNodeStateDescriptors,
     &   FNST => gar_fluidNodeStatuses,
     &   TF => gar_fluidNodeTemps,
     &   FT => gar_fluidNodeFluidTypes,
     &   FCUD => gar_fluidNodeUserEntityValsChar,
     &   FIUD => gar_fluidNodeUserEntityValsInt,
     &   FRUD => gar_fluidNodeUserEntityValsReal,
     &   QUAL => gar_fluidNodeVapourQualities,
     &   WVSNK => gar_fluidNodeWaterVapourSinkRates,
     &   VOL => gar_fluidNodeVolumes,
     &   VDT => gar_fluidNodeVolumeChangeRates,
     &   VCF => gar_fluidNodeVolumeControlFlags,
     &   FF => gar_fluidNodeWallRoughnesses
      USE gar_GlobalArrays_m,
     &   GFFLG => gar_fluidicCondActs,
     &   LIST71 => gar_fluidicCondNodePairs,
     &   GF => gar_fluidicConds,
     &   GLFLG => gar_linearCondActs,
     &   LIST69 => gar_linearCondNodePairs,
     &   GL => gar_linearConds,
     &   GRFLG => gar_radiativeCondActs,
     &   LIST70 => gar_radiativeCondNodePairs,
     &   GR => gar_radiativeConds,
     &   GVFLG => gar_vfacCondActs,
     &   VLNK => gar_vfacCondNodePairs,
     &   GV => gar_vfacConds,
     &   PHSON => gar_massFlowLinkPhaseSepFlags,
     &   MFLH => gar_massFlowLinkEnthalpies,
     &   QEVL => gar_massFlowLinkEvapHeatFluxes,
     &   GP => gar_massFlowLinkFittingLosses,
     &   M => gar_massFlowLinkFlowRates,
     &   GM => gar_massFlowLinkConds,
     &   FNP => gar_massFlowLinkCondNodePairs,
     &   FLFLG => gar_massFlowLinkStatuses,
     &   PSRC => gar_massFlowLinkPressureSources,
     &   DPSDV => gar_massFlowLinkPSourceDerivs,
     &   MTYPE => gar_massFlowLinkTypes,
     &   MFLX => gar_massFlowLinkVapQuals,
     &   PHSX => gar_massFlowLinkPhaseSepVapQuals
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      COMMON /FLAGS/  FLG    (          41 )
      COMMON /GLOC/   OPBLOK, FORMAT, LINTYP, MODULE,  TFORM,
     $         THEAD, PARNAM, HEADER, TFORMF, SOLTYP, RESULT,
     $          MFFB, TIMELA, OUTIME, SOLVER, QTRSOL
      COMMON /GLOI/   IG     (          25 )
      COMMON /GLOR/   RG     (          49 )
      COMMON /LOCC/   CL     (           1 )
      COMMON /LOCI/   IL     (           2 )
      COMMON /LOCR/   RL     (           1 )
      COMMON /SPRNOD/ SPRNDE (           1 ,           3 )
      COMMON /SPRSIZ/ SPRNDM, SPRNDN, USRNDC, USRNDI, USRNDR
      CHARACTER ACGET *132
      CHARACTER CL *1
      CHARACTER FORMAT *8
      CHARACTER FTYPEC *24
      CHARACTER GETFRG *4, GETFST *4, GETFT *24, GETL *24
      CHARACTER HEADER *132
      CHARACTER LINTYP *6
      CHARACTER MFFB *3, MODULE *6
      CHARACTER OPBLOK *10, OUTIME *18
      CHARACTER PARNAM *24
      CHARACTER QTRSOL *3
      CHARACTER RESULT *80
      CHARACTER SOLTYP *7, SOLVER *6, SPRNDE *18, STATRP *3
      CHARACTER SUBMDN  *256, SUBMOD *256
      CHARACTER TFORM *132, TFORMF *132, THEAD *132, TIMELA *3
      CHARACTER ZDAYDT *17, ZDAYTM *8
      INTEGER ADIM, ADIMVL, AIGET, ASIZE, AUNDF
      INTEGER FLG, FTYPEI
      INTEGER IG, IL, INTNOD, IQQMAX
      INTEGER MATDTI, MATSMI
      INTEGER NODNUM
      INTEGER SPRNDM, SPRNDN, STRLNA
      INTEGER USRNDC, USRNDI,USRNDR
      INTEGER WORST
      DOUBLE PRECISION ACLOSS
      DOUBLE PRECISION FLUXF, FLUXGF, FLUXGL, FLUXGR FLUXGT, FLUXL
      DOUBLE PRECISION FLUXMF, FLUXML, FLUXMR, FLUXR, FLUXT
      DOUBLE PRECISION GRPAVE, GRPMAX, GRPMIN
      DOUBLE PRECISION GRPSUM
      DOUBLE PRECISION HCAP, HTCOEF
      DOUBLE PRECISION INTCY1, INTCY2, INTCY3, INTCYC, INTEGL
      DOUBLE PRECISION INTERP, INTGL1, INTGL2, INTRP1, INTRP2, INTRP3
      DOUBLE PRECISION INTRPA
      DOUBLE PRECISION LSTRP1, LSTRP2
      DOUBLE PRECISION MATDTR, MATSMR
      DOUBLE PRECISION NDMFL, NODFN1, NODFN2, NUVRE
      DOUBLE PRECISION POLYNM
      DOUBLE PRECISION RG, RL
      DOUBLE PRECISION SRLMAX, STVRE
      LOGICAL AFTER, AT
      LOGICAL BEFORE, BTWEEN
      LOGICAL SUNDCK
      EXTERNAL GETA
      EXTERNAL GETALP
      EXTERNAL GETC
      EXTERNAL GETCMP
      EXTERNAL GETCCR
      EXTERNAL GETEPS
      EXTERNAL GETFD
      EXTERNAL GETFE
      EXTERNAL GETFF
      EXTERNAL GETFH
      EXTERNAL GETFL
      EXTERNAL GETFLA
      EXTERNAL GETFM
      EXTERNAL GETFQ
      EXTERNAL GETFR
      EXTERNAL GETFRG
      EXTERNAL GETFST
      EXTERNAL GETFT
      EXTERNAL GETFW
      EXTERNAL GETFX
      EXTERNAL GETFY
      EXTERNAL GETFZ
      EXTERNAL GETGL
      EXTERNAL GETGL2
      EXTERNAL GETGF
      EXTERNAL GETGF2
      EXTERNAL GETGP
      EXTERNAL GETGR
      EXTERNAL GETGR2
      EXTERNAL GETGV
      EXTERNAL GETGV2
      EXTERNAL GETL
      EXTERNAL GETM
      EXTERNAL GETP
      EXTERNAL GETPHI
      EXTERNAL GETQA
      EXTERNAL GETQAI
      EXTERNAL GETQE
      EXTERNAL GETQEI
      EXTERNAL GETQI
      EXTERNAL GETQR
      EXTERNAL GETQS
      EXTERNAL GETQSI
      EXTERNAL GETT
      EXTERNAL GETVDT
      EXTERNAL GETVOL
      EXTERNAL GETVQ
      EXTERNAL SETA
      EXTERNAL SETALP
      EXTERNAL SETC
      EXTERNAL SETCMP
      EXTERNAL SETEPS
      EXTERNAL SETFD
      EXTERNAL SETFE
      EXTERNAL SETFF
      EXTERNAL SETFLA
      EXTERNAL SETFH
      EXTERNAL SETFL
      EXTERNAL SETFM
      EXTERNAL SETFQ
      EXTERNAL SETFR
      EXTERNAL SETFRG
      EXTERNAL SETFST
      EXTERNAL SETFT
      EXTERNAL SETFW
      EXTERNAL SETFX
      EXTERNAL SETFY
      EXTERNAL SETFZ
      EXTERNAL SETGL
      EXTERNAL SETGL2
      EXTERNAL SETGF
      EXTERNAL SETGF2
      EXTERNAL SETGP
      EXTERNAL SETGR
      EXTERNAL SETGR2
      EXTERNAL SETGV
      EXTERNAL SETGV2
      EXTERNAL SETL
      EXTERNAL SETM
      EXTERNAL SETP
      EXTERNAL SETPHI
      EXTERNAL SETQA
      EXTERNAL SETQAI
      EXTERNAL SETQE
      EXTERNAL SETQEI
      EXTERNAL SETQI
      EXTERNAL SETQR
      EXTERNAL SETQS
      EXTERNAL SETQSI
      EXTERNAL SETT
      EXTERNAL SETVDT
      EXTERNAL SETVOL
      EXTERNAL SETVQ
      PARAMETER (IQQMAX = 999999)
