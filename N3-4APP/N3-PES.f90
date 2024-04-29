module rep_ker
implicit none
real*8, parameter :: dk26f1 = 1.0d0/14.0d0, dk26f2 = 1.0d0/18.0d0

contains

function drker26(x,xi)
implicit none
real*8, intent(in) :: x, xi
real*8 :: drker26, xl, xs

xl = x
xs = xi
if (x .lt. xi) then
  xl = xi
  xs = x
end if

drker26 = dk26f1/xl**7 - dk26f2*xs/xl**8

end function drker26

function ddrker26(x,xi)
implicit none
real*8, intent(in) :: x, xi
real*8 :: ddrker26, xl, xs

if (x .lt. xi) then
  ddrker26 = -dk26f2/xi**8
else
  ddrker26 = -7.0d0*dk26f1/x**8 + 8.0d0*dk26f2*xi/x**9
end if

end function ddrker26

end module rep_ker

module surface1
implicit none
real*8, allocatable, dimension(:,:) :: asy_array1, asy_array2, asy_array3, &
darray1, darray2, darray3
integer :: na1, na2, na3, nda1, nda2, nda3

contains

subroutine pes3d(r12,r23,r31,totener,dvdr)
implicit none
real*8,parameter :: m1= 14.003074d0, m2 = m1,  m3 = m1 
real*8, intent(in) :: r12, r23, r31
real*8, dimension(:), intent(out) :: dvdr(3)
real*8, intent(out) :: totener
real*8 :: capr, theta
real*8, dimension(:,:) :: derv(3,3), deri(3,3), dw(3,3)
real*8, dimension(:) :: derj(3), w(3), ener(3), r(3), m(3)

!      3
!      C
!     / \
!  r3/   \r2
!   /     \
!  /       \
! O---------O
! 1   r1    2

!=====================================================
!
!PES1                     PES2            PES3
!        3             3                 3
!        C             C                 C
!  r3   /               \               /
!      /  r2             \r1        r1 /th r2
!     /th             th /\           /\  
!O___/_____O         r2 /  \         /  \
!1   r1    2           /    O      1O    \
!                     / r3  2         r3  \
!                   1O                    O2
!

    r(1)=r12
    r(2)=r31
    r(3)=r23
    m(1) = m2
    m(2) = m1
    m(3) = m3
    call crdtrf(r,m,capr,theta,derv)
    call calcener(capr,r12,theta, ener(1), derj, 1)
    deri(1,1) = derj(1)*derv(1,1) + derj(2)*derv(2,1) + derj(3)*derv(3,1)
    deri(1,3) = derj(1)*derv(1,2) + derj(2)*derv(2,2) + derj(3)*derv(3,2)
    deri(1,2) = derj(1)*derv(1,3) + derj(2)*derv(2,3) + derj(3)*derv(3,3)

    r(1)=r23
    r(2)=r31
    r(3)=r12
    m(1) = m2
    m(2) = m3
    m(3) = m1
    call crdtrf(r,m,capr,theta,derv)
    call calcener(capr,r23,theta, ener(2), derj, 2)
!    ener(2)=ener(2)+0.07229929947d0
!    ener(2)=ener(2)+0.07728858179709341d0

    deri(2,2) = derj(1)*derv(1,1) + derj(2)*derv(2,1) + derj(3)*derv(3,1)
    deri(2,3) = derj(1)*derv(1,2) + derj(2)*derv(2,2) + derj(3)*derv(3,2)
    deri(2,1) = derj(1)*derv(1,3) + derj(2)*derv(2,3) + derj(3)*derv(3,3)

    r(1)=r31
    r(2)=r23
    r(3)=r12
    m(1) = m1
    m(2) = m3
    m(3) = m2
    call crdtrf(r,m,capr,theta,derv)
    call calcener(capr,r31,theta, ener(3), derj, 3)
!    ener(3)=ener(3)+0.07229929947d0
!    ener(3)=ener(3)+0.07728858179709341d0

    deri(3,3) = derj(1)*derv(1,1) + derj(2)*derv(2,1) + derj(3)*derv(3,1)
    deri(3,2) = derj(1)*derv(1,2) + derj(2)*derv(2,2) + derj(3)*derv(3,2)
    deri(3,1) = derj(1)*derv(1,3) + derj(2)*derv(2,3) + derj(3)*derv(3,3)

    call weights(r12, r23, r31, w, dw)
    totener = sum(ener*w)
!    totener=ener(2)

    dvdr(1) = sum(deri(:,1)*w)+sum(dw(:,1)*ener)
    dvdr(2) = sum(deri(:,2)*w)+sum(dw(:,2)*ener)
    dvdr(3) = sum(deri(:,3)*w)+sum(dw(:,3)*ener)

end subroutine pes3d

subroutine crdtrf(r,m,capr,theta, derv)
implicit none
real*8, dimension(:), intent(in) :: r(3), m(3)
real*8, dimension(:,:), intent(out) :: derv(3,3)
real*8, intent(out) :: capr, theta
real*8, parameter :: pc = sqrt(epsilon(1.0d0))
real*8 :: cmu1, cmu2, rcm1, rcm2

!        3
!       /|\
!      / | \
!   r3/ R|  \r2
!    /   |th \
!   /____|____\ 
!  1    r1    2

cmu1 = m(2) / (m(1)+m(2))
cmu2 = m(1) / (m(1)+m(2))

rcm1 = r(1) * cmu1
rcm2 = r(1) * cmu2

capr = sqrt (r(2)**2/r(1)*rcm1 + r(3)**2/r(1)*rcm2 - rcm1*rcm2 )
if (abs(capr) < pc) capr = pc

theta = (rcm2**2+capr**2-r(2)**2)/2.0d0/rcm2/capr
if (theta > 1.0d0) theta = 1.0d0
if (theta < -1.0d0) theta = -1.0d0
theta = acos(theta)

derv(1,1) = -cmu1*cmu2*r(1)/capr

derv(1,2) = r(2)*cmu1/capr

derv(1,3) = r(3)*cmu2/capr

derv(2,1) = 1.0d0
derv(2,2) = 0.0d0
derv(2,3) = 0.0d0

derv(3,1) = (derv(1,1)/capr*cos(theta)+cos(theta)/r(1)-(capr*derv(1,1)+rcm2*cmu2)/rcm2/capr)&
            /sqrt(1.0d0-cos(theta)**2)

derv(3,2) = (r(2)/rcm2/capr-derv(1,2)/rcm2+cos(theta)/capr*derv(1,2))/sqrt(1.0d0-cos(theta)**2)

derv(3,3) = (cos(theta)/capr*derv(1,3)-derv(1,3)/rcm2)/sqrt(1.0d0-cos(theta)**2)

return

end subroutine crdtrf

subroutine weights(d1,d2,d3,w,dw)
implicit none
real*8, intent(in) :: d1, d2, d3
real*8, dimension(:), intent(out) :: w(3)
real*8, dimension(:,:), intent(out) :: dw(3,3)
integer, parameter :: power = 2
!real*8, parameter :: wtol = epsilon(1.0d0), dr1 = 1.10d0, dr2 = 1.05d0, dr3 =1.05d0
real*8, parameter :: wtol = epsilon(1.0d0), dr1 = 1.00d0, dr2 = 1.00d0, dr3 =1.00d0
real*8 :: wsum, pw1, pw2, pw3, r1, r2, r3

r1=d1
r2=d2
r3=d3

wsum = 0.0d0
do while(wsum < wtol)
  pw1 = exp(-(r1/dr1)**power)
  pw2 = exp(-(r2/dr2)**power)
  pw3 = exp(-(r3/dr3)**power)
  wsum = pw1+pw2+pw3
  if(wsum < wtol) then
    r1=r1-0.1d0
    r2=r2-0.1d0
    r3=r3-0.1d0
  end if
end do

  w(1) = pw1 / wsum
  w(2) = pw2 / wsum
  w(3) = pw3 / wsum

  dw(1,1) =-power*((r1/dr1)**(power-1))*(pw1*pw2+pw1*pw3)/dr1/wsum**2
  dw(1,2) = power*((r2/dr2)**(power-1))*pw1*pw2/dr2/wsum**2
  dw(1,3) = power*((r3/dr3)**(power-1))*pw1*pw3/dr3/wsum**2

  dw(2,1) = power*((r1/dr1)**(power-1))*pw2*pw1/dr1/wsum**2
  dw(2,2) =-power*((r2/dr2)**(power-1))*(pw2*pw1+pw2*pw3)/dr2/wsum**2
  dw(2,3) = power*((r3/dr3)**(power-1))*pw2*pw3/dr3/wsum**2


  dw(3,1) = power*((r1/dr1)**(power-1))*pw3*pw1/dr1/wsum**2
  dw(3,2) = power*((r2/dr2)**(power-1))*pw3*pw2/dr2/wsum**2
  dw(3,3) =-power*((r3/dr3)**(power-1))*(pw3*pw1+pw3*pw2)/dr3/wsum**2

return

end subroutine weights

subroutine calcener(capr,smlr,theta, ener, der, sno)
use rep_ker
use RKHS            ! This module needs to be used by your code
!use param
implicit none
real*8 :: lambda
real*8, intent(out) :: ener
real*8, intent(in) :: capr, smlr, theta
real*8, dimension(:), intent(out) :: der(3)
integer, intent(in) :: sno
real*8,parameter :: pi = acos(-1.0d0), piby180 = pi/180.0d0
real*8 :: asener, anener, asder, z1, z2
real*8, dimension(:) :: ander(3), x(3)
integer :: kk, ii
type(kernel), save  :: pes11, pes12, pes13           ! The kernel type is needed to set up and evaluate a RKHS model
logical, save :: stored = .false., kread = .false.
logical, save :: ker1 = .false., ker2 = .false., ker3 = .false.
character (len=80), save :: datapath="data/"


if (.not. ker1) then
  inquire(file=trim(datapath)//"pes11.kernel", exist=ker1)   ! file_exists will be true if the file exists and false otherwise
end if

if (.not. ker2) then
  inquire(file=trim(datapath)//"pes12.kernel", exist=ker2)   ! file_exists will be true if the file exists and false otherwise
end if

if (.not. ker3) then
  inquire(file=trim(datapath)//"pes13.kernel", exist=ker3)   ! file_exists will be true if the file exists and false otherwise
end if

lambda=0.1d-16

if (.not. stored ) then

open(unit=1001,file=trim(datapath)//"asymp.dat", status = "old")

read(1001,*)na1
allocate(asy_array1(na1,2))
do ii = 1, na1
  read(1001,*)asy_array1(ii,1), asy_array1(ii,2)
end do

read(1001,*)na2
allocate(asy_array2(na2,2))
do ii = 1, na2
  read(1001,*)asy_array2(ii,1), asy_array2(ii,2)
end do

read(1001,*)na3
allocate(asy_array3(na3,2))
do ii = 1, na3
  read(1001,*)asy_array3(ii,1), asy_array3(ii,2)
end do

read(1001,*)nda1
allocate(darray1(nda1,2))
do ii = 1, nda1
  read(1001,*)darray1(ii,1), darray1(ii,2)
end do

read(1001,*)nda2
allocate(darray2(nda2,2))
do ii = 1, nda2
  read(1001,*)darray2(ii,1), darray2(ii,2)
end do

read(1001,*)nda3
allocate(darray3(nda3,2))
do ii = 1, nda3
  read(1001,*)darray3(ii,1), darray3(ii,2)
end do

stored = .true.

end if

if (.not. kread) then
  if (ker1 .and. ker2 .and. ker3 ) then
    call pes11%load_from_file(trim(datapath)//"pes11.kernel")
    call pes12%load_from_file(trim(datapath)//"pes12.kernel")
    call pes13%load_from_file(trim(datapath)//"pes13.kernel")
    kread = .true.
  else
    call pes11%read_grid(trim(datapath)//"pes11.csv")
!    print*,"IAMHERE"
    call pes11%k1d(1)%init(TAYLOR_SPLINE_N2_KERNEL)         ! choose one-dimensional kernel for dimension 1
    call pes11%k1d(2)%init(RECIPROCAL_POWER_N2_M6_KERNEL)   ! choose one-dimensional kernel for dimension 2
    call pes11%k1d(3)%init(RECIPROCAL_POWER_N2_M6_KERNEL) 

!    call pes11%calculate_coefficients_slow(lambda)
    call pes11%calculate_coefficients_fast()

    call pes11%calculate_sums()

    call pes11%save_to_file(trim(datapath)//"pes11.kernel")
!
    call pes12%read_grid(trim(datapath)//"pes12.csv")
!    print*,"IAMHERE"
    call pes12%k1d(1)%init(TAYLOR_SPLINE_N2_KERNEL)         ! choose one-dimensional kernel for dimension 1
    call pes12%k1d(2)%init(RECIPROCAL_POWER_N2_M6_KERNEL)   ! choose one-dimensional kernel for dimension 2
    call pes12%k1d(3)%init(RECIPROCAL_POWER_N2_M6_KERNEL)

!    call pes11%calculate_coefficients_slow(lambda)
    call pes12%calculate_coefficients_fast()

    call pes12%calculate_sums()

    call pes12%save_to_file(trim(datapath)//"pes12.kernel")

    call pes13%read_grid(trim(datapath)//"pes13.csv")
!    print*,"IAMHERE"
    call pes13%k1d(1)%init(TAYLOR_SPLINE_N2_KERNEL)         ! choose one-dimensional kernel for dimension 1
    call pes13%k1d(2)%init(RECIPROCAL_POWER_N2_M6_KERNEL)   ! choose one-dimensional kernel for dimension 2
    call pes13%k1d(3)%init(RECIPROCAL_POWER_N2_M6_KERNEL)

!    call pes13%calculate_coefficients_slow(lambda)
    call pes13%calculate_coefficients_fast()

    call pes13%calculate_sums()

    call pes13%save_to_file(trim(datapath)//"pes13.kernel")

    kread = .true.
  end if
end if

x(1)=(1.0d0-cos(theta))/2.0d0
x(2)=capr
x(3)=smlr

if (sno==1) then
  asener = 0.0d0
  asder=0.0d0
  do kk = 1,na1
    asener = asener + drker26(smlr,asy_array1(kk,1))*asy_array1(kk,2)
    asder = asder + ddrker26(smlr,asy_array1(kk,1))*asy_array1(kk,2)
  end do

  anener = 0.0d0
  ander=0.0d0
  call pes11%evaluate_fast(x,anener,ander)

    ener = asener+anener
    der(1) = ander(2)
    der(2) = asder+ander(3)
    der(3) = ander(1)*sin(theta)/2.0d0
end if

if (sno==2) then
  asener = 0.0d0
  asder=0.0d0
  do kk = 1,na2
    asener = asener + drker26(smlr,asy_array2(kk,1))*asy_array2(kk,2)
    asder = asder + ddrker26(smlr,asy_array2(kk,1))*asy_array2(kk,2)
  end do

  anener = 0.0d0
  ander=0.0d0
  call pes12%evaluate_fast(x,anener,ander)

    ener = asener+anener
    der(1) = ander(2)
    der(2) = asder+ander(3)
    der(3) = ander(1)*sin(theta)/2.0d0
end if

if (sno==3) then
  asener = 0.0d0
  asder=0.0d0
  do kk = 1,na3
    asener = asener + drker26(smlr,asy_array3(kk,1))*asy_array3(kk,2)
    asder = asder + ddrker26(smlr,asy_array3(kk,1))*asy_array3(kk,2)
  end do

  anener = 0.0d0
  ander=0.0d0
  call pes13%evaluate_fast(x,anener,ander)

    ener = asener+anener
    der(1) = ander(2)
    der(2) = asder+ander(3)
    der(3) = ander(1)*sin(theta)/2.0d0
end if

return

end subroutine calcener

end module

module callpot1
use surface1

contains

subroutine n34apppes(tmpr, totener, dvdr)
implicit none
real*8, dimension (:), intent (in) :: tmpr(3)
real*8, dimension (:), intent (out) :: dvdr(3)
real*8, intent (out) :: totener
real*8, dimension (:) :: dvdx(4), xp(3), tmpdvdr(3), r(3)
real*8, parameter :: dx = 0.005d0
real*8 :: d0h, d02h, ener
integer :: ii

!      3
!      C
!     / \
!  r3/   \r2
!   /     \
!  /       \
! O---------O
! 1   r1    2

!C1+O2O3 

r(1)=tmpr(2) !OO
r(2)=tmpr(3) !OC
r(3)=tmpr(1) !CO

call pes3d(r(1),r(2),r(3),totener,dvdr)

if (any(isNaN(dvdr))) then
  do ii = 1, 3 
!==========================================================
! Richardson Extrapolation to calculate first derivative  =
!==========================================================
!-------------------------P1 (x-2h)------------------------
    xp=r
    xp(ii)=r(ii)-2.0d0*dx
    call pes3d(xp(1),xp(2),xp(3),ener,tmpdvdr)
    dvdx(1)=ener

    xp=r
    xp(ii)=r(ii)-dx
    call pes3d(xp(1),xp(2),xp(3),ener,tmpdvdr)
    dvdx(2)=ener

    xp=r
    xp(ii)=r(ii)+dx
    call pes3d(xp(1),xp(2),xp(3),ener,tmpdvdr)
    dvdx(3)=ener

    xp=r
    xp(ii)=r(ii)+2.0d0*dx
    call pes3d(xp(1),xp(2),xp(3),ener,tmpdvdr)
    dvdx(4)=ener

    d0h=(dvdx(3)-dvdx(2))/2.0d0/dx
    d02h=(dvdx(4)-dvdx(1))/4.0d0/dx
    dvdr(ii)=(4.0d0*d0h-d02h)/3.0d0
  end do
end if

return

end subroutine n34apppes

subroutine diatoo(r,ener,der)
use rep_ker
implicit none
real*8, intent(in) :: r
real*8, intent(out) :: ener, der
integer :: kk

ener=0.0d0
der=0.0d0
do kk = 1, nda1
    ener = ener + drker26(r,darray1(kk,1))*darray1(kk,2)
    der = der + ddrker26(r,darray1(kk,1))*darray1(kk,2)
end do

return

end subroutine diatoo

subroutine diatco(r,ener,der)
use rep_ker
implicit none
real*8, intent(in) :: r
real*8, intent(out) :: ener, der
integer :: kk

ener=0.0d0
der=0.0d0
do kk = 1, nda2
    ener = ener + drker26(r,darray2(kk,1))*darray2(kk,2)
    der = der + ddrker26(r,darray2(kk,1))*darray2(kk,2)
end do

return

end subroutine diatco

end module
