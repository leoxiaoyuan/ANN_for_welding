
      SUBROUTINE DFLUX(FLUX,SOL,KSTEP,KINC,TIME,NOEL,NPT,COORDS,JLTYP,TE
     1 MP,PRESS,SNAME)

      INCLUDE 'ABA_PARAM.INC' 

      DIMENSION FLUX(1),TIME(2),COORDS(3)

      CHARACTER*80 SNAME
C Body heat source
      JLTYP=1     

	bead_length = .080
        
      stop_time_at_start = 0
        
	welding_time = 35.24000
        
	arc_speed = .00227
      
	heat_input = 1100
        
	heat_source_size_coeff = 1
      
      F=0

C     Current integral coordinate
      x=COORDS(1)
	y=COORDS(2)
	z=COORDS(3)

C     Double ellipsoid shape parameters
	a1=0.0045 * heat_source_size_coeff
      a2=0.0065 * heat_source_size_coeff
	b=0.0068 * heat_source_size_coeff
	c=0.0026 * heat_source_size_coeff

C     F1 and f2 are the heat source distribution coefficients
      f1=1
      f2=2.0-f1
	PI=3.1415926
      
      IF(TIME(1).LE.(stop_time_at_start))THEN          
      Xnow = - bead_length / 2
      END IF
      IF(TIME(1).GE.(stop_time_at_start).AND.TIME(1).LE.(stop_time_at_st
     1 art + welding_time))THEN       
      Xnow = - bead_length / 2 + arc_speed * (TIME(1) - stop_time_at_sta
     1 rt)
      END IF
      IF(TIME(1).GE.(stop_time_at_start + welding_time))THEN    
      Xnow = bead_length / 2 
      END IF
      
C     Function Definition
      heat1=6.0*sqrt(3.0)*heat_input/(a1*b*c*PI*sqrt(PI))*f1
      heat2=6.0*sqrt(3.0)*heat_input/(a2*b*c*PI*sqrt(PI))*f2
	shape1=exp(-3.0*(x-Xnow)**2/a1**2-3.0*(y-0)**2/b**2
     1 -3.0*(z-0)**2/c**2)
      shape2=exp(-3.0*(x-Xnow)**2/a2**2-3.0*(y-0)**2/b**2
     1 -3.0*(z-0)**2/c**2)      
      
	if (x .GE. Xnow) then
		FLUX(1)=heat1*shape1
      else
          FLUX(1)=heat2*shape2
	endif

      RETURN

      END

