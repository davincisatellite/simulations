      PROGRAM FINALMODEL_SCHEDULEPERIO                                          
C   
      INCLUDE 'FINALMOD.h'                                                      
C   
      CHARACTER MNAME * 24                                                      
C   
      MNAME = 'FINALMODEL_SCHEDULEPERIO'                                        
C   
      FLG(1) = 430                                                              
      FLG(2) = 914                                                              
      FLG(3) = 0                                                                
      FLG(4) = 17795                                                            
      FLG(5) = 0                                                                
      FLG(6) = 1                                                                
      FLG(7) = 0                                                                
      FLG(8) = 0                                                                
      FLG(9) = 0                                                                
      FLG(10) = 0                                                               
      FLG(11) = 2                                                               
      FLG(12) = 0                                                               
      FLG(13) = 49                                                              
      FLG(14) = 25                                                              
      FLG(15) = 16                                                              
      FLG(16) = 1                                                               
      FLG(17) = 92                                                              
      FLG(18) = 66                                                              
      FLG(19) = 6                                                               
      FLG(20) = 0                                                               
      FLG(21) = 0                                                               
      FLG(22) = 0                                                               
      FLG(23) = 0                                                               
      FLG(24) = 77438                                                           
      FLG(25) = 2                                                               
      FLG(26) = 1                                                               
      FLG(27) = 133                                                             
      FLG(28) = 1024729                                                         
      FLG(29) = 1                                                               
      FLG(30) = 0                                                               
      FLG(31) = 0                                                               
      FLG(32) = 0                                                               
      FLG(33) = 0                                                               
      FLG(34) = 0                                                               
      FLG(35) = 22                                                              
      FLG(36) = 1                                                               
      FLG(37) = 0                                                               
      FLG(38) = 0                                                               
      FLG(39) = 35590                                                           
      FLG(40) = 0                                                               
      FLG(41) = 0                                                               
      CALL SVMNAM(MNAME)                                                        
C   
      SPRNDM = 1                                                                
      SPRNDN = 3                                                                
      USRNDC = 0                                                                
      USRNDI = 0                                                                
      USRNDR = 0                                                                
C   
      CALL MAINA(MNAME)                                                         
C   
      CALL SUCCES                                                               
C   
      STOP                                                                      
C   
      END                                                                       
      SUBROUTINE IN0001                                                 
      INCLUDE 'FINALMOD.h'
      LOGICAL HTFLAG                                                    
      HTFLAG = .TRUE.                                                   
      OPBLOK = 'INITIAL   '                                             
      IG(2) = 1                                                         
      CALL FINITS(' ' , -    1)                                         
      OPBLOK = ' '                                                      
      RETURN                                                            
      END                                                               
      SUBROUTINE V10001                                                 
      INCLUDE 'FINALMOD.h'
      LOGICAL HTFLAG                                                    
      HTFLAG = (SOLTYP .EQ. 'THERMAL' .OR. SOLTYP .EQ. 'HUMID')         
      OPBLOK = 'VARIABLES1'                                             
      IG(2) = 1                                                         
      CALL PRMUPD                                                       
      CALL CHKTRM                                                       
      CALL ACDDYU                                                       
      CALL HEATER_UPDATE                                                
      IG(2) = 1                                                         
      QI(MD(3,1)+95)=0.0872676*INTRP1(RG(16),MD(14,1)+4,0)              
      QI(MD(3,1)+96)=0.0872676*INTRP1(RG(16),MD(14,1)+4,0)              
      QI(MD(3,1)+97)=0.0872676*INTRP1(RG(16),MD(14,1)+4,0)              
      QI(MD(3,1)+98)=0.0872676*INTRP1(RG(16),MD(14,1)+4,0)              
      QI(MD(3,1)+99)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)              
      QI(MD(3,1)+100)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+101)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+102)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+103)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+104)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+105)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+106)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+107)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+108)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+109)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+110)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+111)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+112)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+113)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+114)=0.0188662*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+115)=0.0872676*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+116)=0.0872676*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+117)=0.0872676*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+118)=0.0872676*INTRP1(RG(16),MD(14,1)+4,0)             
      QI(MD(3,1)+213)=0.0379747*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+214)=0.0379747*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+215)=0.0379747*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+216)=0.0379747*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+217)=0.0632911*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+218)=0.0632911*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+219)=0.0632911*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+220)=0.0632911*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+221)=0.0237342*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+222)=0.0237342*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+223)=0.0237342*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+224)=0.0237342*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+225)=0.0632911*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+226)=0.0632911*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+227)=0.0632911*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+228)=0.0632911*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+229)=0.0237342*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+230)=0.0237342*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+231)=0.0237342*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+232)=0.0237342*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+233)=0.0379747*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+234)=0.0379747*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+235)=0.0379747*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+236)=0.0379747*INTRP1(RG(16),MD(14,1)+2,0)             
      QI(MD(3,1)+23)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+24)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+25)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+26)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+27)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+28)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+29)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+30)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+31)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+32)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+33)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+34)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+35)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+36)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+37)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+38)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+39)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+40)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+41)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+42)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+43)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+44)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+45)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+46)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+47)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+48)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+49)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+50)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+51)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+52)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+53)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+54)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+55)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+56)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+57)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+58)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+59)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+60)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+61)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+62)=0.00944767*INTRP1(RG(16),MD(14,1)+1,0)             
      QI(MD(3,1)+63)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+64)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+65)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+66)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+67)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+68)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+69)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+70)=0.0436047*INTRP1(RG(16),MD(14,1)+1,0)              
      QI(MD(3,1)+167)=0.0379747*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+168)=0.0379747*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+169)=0.0379747*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+170)=0.0379747*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+171)=0.0632911*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+172)=0.0632911*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+173)=0.0632911*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+174)=0.0632911*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+175)=0.0237342*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+176)=0.0237342*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+177)=0.0237342*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+178)=0.0237342*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+179)=0.0632911*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+180)=0.0632911*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+181)=0.0632911*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+182)=0.0632911*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+183)=0.0237342*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+184)=0.0237342*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+185)=0.0237342*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+186)=0.0237342*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+187)=0.0379747*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+188)=0.0379747*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+189)=0.0379747*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+190)=0.0379747*INTRP1(RG(16),MD(14,1)+5,0)             
      QI(MD(3,1)+143)=0.0723684*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+144)=0.0723684*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+145)=0.0723684*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+146)=0.0723684*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+147)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+148)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+149)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+150)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+151)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+152)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+153)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+154)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+155)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+156)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+157)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+158)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+159)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+160)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+161)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+162)=0.0263158*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+163)=0.0723684*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+164)=0.0723684*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+165)=0.0723684*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+166)=0.0723684*INTRP1(RG(16),MD(14,1)+3,0)             
      QI(MD(3,1)+71)=0.0946970*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+72)=0.0946970*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+73)=0.0946970*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+74)=0.0946970*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+75)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+76)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+77)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+78)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+79)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+80)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+81)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+82)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+83)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+84)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+85)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+86)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+87)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+88)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+89)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+90)=0.0151515*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+91)=0.0946970*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+92)=0.0946970*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+93)=0.0946970*INTRP1(RG(16),MD(14,1)+6,0)              
      QI(MD(3,1)+94)=0.0946970*INTRP1(RG(16),MD(14,1)+6,0)              
      CALL GM0001(HTFLAG)                                               
      OPBLOK = ' '                                                      
      RETURN                                                            
      END                                                               
      SUBROUTINE GM0001(HTFLAG)                                         
      LOGICAL HTFLAG                                                    
      RETURN                                                            
      END                                                               
      SUBROUTINE V20001                                                 
      INCLUDE 'FINALMOD.h'
      OPBLOK = 'VARIABLES2'                                             
      IG(2) = 1                                                         
      CALL SSNCNT(FLG(24),FLG(25),MAX0(FLG(1),1),PCS,T)                 
      IG(25) = IG(25) + 1                                               
      CALL PRMUPD                                                       
      CALL HEATER_UPDATE                                                
      IG(2) = 1                                                         
      OPBLOK = ' '                                                      
      CALL PARWRT('VARIABLES2')                                         
      RETURN                                                            
      END                                                               
      SUBROUTINE EXECTN                                                 
      INCLUDE 'FINALMOD.h'
      RG(13)=0.01                                                       
      IG(4)=100                                                         
      CALL SOLVFM                                                       
      RG(18)=RU(MD(8,1)+1)                                              
      RG(3)=RG(18)/100.0                                                
      RG(12)=RG(18)/10.0                                                
      CALL SLCRNC                                                       
      RG(18)=5736.629733557936                                          
      RG(12)=573.6629733557936                                          
      IG(4)=100                                                         
      RG(13)=0.01                                                       
      RG(3)=100.0                                                       
      CALL SLCRNC                                                       
      RG(18)=5736.629733557936                                          
      RG(12)=573.6629733557936                                          
      IG(4)=100                                                         
      RG(13)=0.01                                                       
      RG(3)=100.0                                                       
      CALL SOLCYC('SLCRNC',0.01D0,0.01D0,5736.629733557936D0,10,' ','NON
     $E')                                                               
      CALL SLCRNC                                                       
      IG(4)=100                                                         
      RG(13)=0.01                                                       
      CALL SOLVFM                                                       
      IG(4)=100                                                         
      RG(13)=0.01                                                       
      CALL SOLVFM                                                       
      RETURN                                                            
      END                                                               
      SUBROUTINE OUTPUT                                                 
      INCLUDE 'FINALMOD.h'
      IF (OUTIME .NE. 'ALL') RETURN                                     
      OPBLOK = 'OUTPUTS'                                                
      CALL PRNDTB(' ','L, T, QS, QE, QA, QI, C',1)                      
      CALL DMPTMD(' ','NODES, CONDUCTORS',1,' ')                        
      OPBLOK = ' '                                                      
      CALL PARWRT('OUTPUTS')                                            
      RETURN                                                            
      END                                                               
