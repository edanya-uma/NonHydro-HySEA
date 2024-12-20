      SUBROUTINE PRTOXY ( ALATDG, ALNGDG, ALATO, ALNGO, X, Y, IND )
      
      implicit none
      
      real*8 A,E2,E12,D,RD,RLAT,ALATDG,SLAT,V2,CLAT,AL,RPH1
      real*8 ALNGDG,ALATO,ALNGO,X,Y,SRPH1,RPH2,SRPH2,R,AN,C1,C2
      real*8 RLATO,SLATO,CLATO,DEN,TPH1,BL,PH1,CPH1
      integer IND
      
      
c      C*************************************************************C
c      C*****                                                   *****C
c      C***** Conversion between (lat,lon) and (X,Y)            *****C
c      C*****      using Gauss-Krueger projection               *****C
c      C*****                                                   *****C
c      C*************************************************************C
      
c      C***** Input/Output
c      C***** ALATDG,ALNGDG : latitude, longitude (deg)
c      C***** X , Y : EW, NS coordinates (km)
      
c      C***** Input
c      C***** ALATO,ALNGO : origin of projection (deg)
c      C***** IND : =0 ... transform (lat,lng) to ( X , Y )
c      C***** : =1 ... transform ( X , Y ) to (lat,lng)
      
      parameter ( A=6378.160, E2=6.6946053E-3, E12=6.7397251E-3 )
      parameter ( D=57.29578, RD=1./57.29578 )
      
c      C------------------------------------------------
c      C----- IND=0 : transform (lat,lng) to (X,Y) -----
c      C------------------------------------------------
      
      IF (IND .EQ. 0)  THEN
	RLAT=ALATDG*RD
	SLAT=SIN(RLAT)
	CLAT=COS(RLAT)
	V2=1.+E12*CLAT*CLAT
	AL=ALNGDG-ALNGO
	PH1=ALATDG+0.5*V2*AL*AL*SLAT*CLAT*RD
	RPH1=PH1*RD
	RPH2=(PH1+ALATO)/2.*RD
	SRPH1=SIN(RPH1)
	SRPH2=SIN(RPH2)
c      C-----
        
	R  = A*(1. - E2) / SQRT( 1. - E2*SRPH2*SRPH2 )**3
	AN = A / SQRT( 1. - E2*SRPH1*SRPH1 )
	C1 = D / R
	C2 = D / AN
	Y =(PH1-ALATO)/C1
	X  = AL*CLAT/C2*( 1. + AL*AL*COS(2.*RLAT)/(6.*D*D) )

c      C------------------------------------------------
c      C----- IND=1 : transform (X,Y) to (LAT,LNG) -----
c      C------------------------------------------------
       	
      ELSEIF(IND .EQ. 1)  THEN
        
	RLATO = ALATO*RD
	SLATO = SIN( RLATO )
	CLATO = COS( RLATO )
	DEN = SQRT( 1. - E2*SLATO*SLATO )
	R =A*(1.-E2)/DEN**3
	AN = A / DEN
	V2 = 1. + E12*CLATO*CLATO
c      C-----
         
	C1 = D / R
	C2 = D / AN
	PH1  = ALATO + C1*Y
	RPH1 = PH1*RD
	TPH1 = TAN(RPH1)
	CPH1 = COS(RPH1)
	BL   = C2*X
	ALATDG = PH1 - 0.5*BL*BL*V2*TPH1*RD
	ALNGDG = ALNGO+BL/CPH1*(1.- BL*BL*(1.+2.*TPH1*TPH1)/(6.*D*D))
      ENDIF
      
      RETURN
      END

