(* Independent compactified background, odd transfer, threshold, and inner checks. *)
ClearAll["Global`*"];
$MaxExtraPrecision = 20000;
wp = 80;
arguments = Rest[$ScriptCommandLine];
outDir = If[Length[arguments] >= 1, First[arguments], FileNameJoin[{Directory[], "build", "wolfram"}]];
If[!DirectoryQ[outDir], CreateDirectory[outDir, CreateIntermediateDirectories -> True]];
python = Import[FileNameJoin[{Directory[], "data", "backgrounds.json"}], "RawJSON"];
targetText = "1e-6";
targetMachine = ToExpression[StringReplace[targetText, "e" -> "*^"]];
anchor = First@MinimalBy[python["records"], Abs[Log[#["target_delta"]/targetMachine]] &];
target = SetPrecision[targetMachine, wp];
qGuess = SetPrecision[anchor["charge"], wp];
pGuess = SetPrecision[anchor["phi_h"], wp];
xStop = SetPrecision[anchor["sigma_k"], wp];

nuLegendre = SetPrecision[-1/2 + I Sqrt[3]/2, wp];
aCritical = a /. FindRoot[
  Re[LegendreP[nuLegendre, (1 + a)/(1 - a)]] == 0,
  {a, SetPrecision[0.80, wp]}, WorkingPrecision -> wp,
  AccuracyGoal -> 60, PrecisionGoal -> 60];
qHCritical = Sqrt[aCritical];
qCritical = 2 qHCritical/(1 + aCritical);

zfun[p_] := Exp[p^2];
dInvZ[p_] := -2 p Exp[-p^2];
ddInvZ[p_] := (-2 + 4 p^2) Exp[-p^2];
eps = SetPrecision[10^-18, wp];
u0 = 1 - eps;

ClearAll[exterior];
exterior[q_?NumericQ, p_?NumericQ] := Module[
  {zH, nPrime, pR, wH, nValue, massU, chiU, nU},
  zH = zfun[p];
  nPrime = 1 - q^2/zH;
  pR = q^2 dInvZ[p]/(2 nPrime);
  wH = -pR;
  nValue = 1 - 2 mass[u] u;
  massU = -nValue w[u]^2/2 - q^2/(2 zfun[scalar[u]]);
  chiU = u w[u]^2;
  nU = -2 mass[u] - 2 u massU;
  NDSolveValue[
    {
      mass'[u] == massU,
      scalar'[u] == w[u],
      w'[u] == (q^2 dInvZ[scalar[u]]/2 -
          (nU - nValue chiU) w[u])/nValue,
      chi'[u] == chiU,
      mass[u0] == 1/2 + q^2 eps/(2 zH),
      scalar[u0] == p - wH eps,
      w[u0] == wH,
      chi[u0] == -wH^2 eps
    },
    {mass, scalar, w, chi}, {u, u0, 0},
    WorkingPrecision -> wp, AccuracyGoal -> 46, PrecisionGoal -> 46,
    MaxSteps -> Infinity, Method -> {"StiffnessSwitching"}
  ]
];

ClearAll[residual, sourceResidual, deltaResidual];
residual[q_?NumericQ, p_?NumericQ] := residual[q, p] = Module[{sol = exterior[q, p]},
  {sol[[2]][0], q/sol[[1]][0]/qCritical - 1 - target}
];
sourceResidual[q_?NumericQ, p_?NumericQ] := First[residual[q, p]];
deltaResidual[q_?NumericQ, p_?NumericQ] := Last[residual[q, p]];

solutionRoot = FindRoot[
  {sourceResidual[q, p] == 0, deltaResidual[q, p] == 0},
  {{q, qGuess}, {p, pGuess}}, WorkingPrecision -> wp,
  AccuracyGoal -> 38, PrecisionGoal -> 38, MaxIterations -> 30];
qRoot = q /. solutionRoot;
pRoot = p /. solutionRoot;
ext = exterior[qRoot, pRoot];
mInfinity = ext[[1]][0];
chiInfinity = ext[[4]][0];
scalarCharge = ext[[3]][0];
measured = qRoot/mInfinity/qCritical - 1;

(* Independent interior continuation, evaluated on the same geometrically
   recorded matching coordinate for the cross-engine anchor. *)
chiH = -chiInfinity;
zH = zfun[pRoot];
hPrimeH = Exp[-chiH] (-1 + qRoot^2/zH);
phiZH = Exp[-chiH] qRoot^2 dInvZ[pRoot]/(2 hPrimeH);
epsI = SetPrecision[10^-10, wp];
x0 = Log[1 + epsI];
z0 = Exp[x0];
bracket0 = -1 + qRoot^2/zH;
bracket1 = 2 + qRoot^2 dInvZ[pRoot] phiZH;
hSecond = Exp[-chiH] (bracket1 - phiZH^2 bracket0)/2;
electricFirst = Exp[-chiH] qRoot^2 ((1 - phiZH^2) dInvZ[pRoot] +
    ddInvZ[pRoot] phiZH)/2;
betaFirst = (electricFirst - 2 hSecond phiZH)/(2 hPrimeH);
phiSecond = (betaFirst - phiZH)/2;
chiSecond = (2 phiZH betaFirst - phiZH^2)/2;
interior = NDSolveValue[
  {
    ph'[x] == beta[x],
    beta'[x] == -(hh'[x]/hh[x]) beta[x] +
      Exp[x] Exp[-ch[x]] qRoot^2 dInvZ[ph[x]]/(2 hh[x]),
    ch'[x] == beta[x]^2,
    hh'[x] == Exp[-ch[x]] (-Exp[-x] + qRoot^2 Exp[x]/zfun[ph[x]]),
    ph[x0] == pRoot + phiZH epsI + phiSecond epsI^2,
    beta[x0] == phiZH + betaFirst epsI,
    ch[x0] == chiH + phiZH^2 epsI + chiSecond epsI^2,
    hh[x0] == hPrimeH epsI + hSecond epsI^2
  },
  {ph, beta, ch, hh}, {x, x0, xStop}, WorkingPrecision -> wp,
  AccuracyGoal -> 38, PrecisionGoal -> 38, MaxSteps -> Infinity,
  MaxStepSize -> SetPrecision[10^-4, wp],
  Method -> {"ExplicitRungeKutta", "DifferenceOrder" -> 8}
];


lambda = 6;
frequencies = SetPrecision[{19/100, 1/5, 21/100}, wp];
xTransferStart = x0;
tTransferStart = Log[xTransferStart];
tTransferStop = Log[xStop];

ClearAll[potential, dxdrStar];
potential[x_?NumericQ] := Module[
  {p, b, c, hv, zz, zv, emc, hx, bx, ar2, s1, s1x, s2, s2x,
   g, v11, v22, v12},
  p = interior[[1]][x]; b = interior[[2]][x];
  c = interior[[3]][x]; hv = interior[[4]][x];
  zz = Exp[x]; zv = zfun[p]; emc = Exp[-c];
  hx = emc (-1/zz + qRoot^2 zz/zv);
  bx = -(hx/hv) b + zz emc qRoot^2 dInvZ[p]/(2 hv);
  ar2 = hv zz^3 emc;
  s1 = hv zz^2;
  s1x = zz^2 (hx + 2 hv);
  s2 = hv zz^2 b (2 p)/2;
  s2x = zz^2 ((hx + 2 hv) b (2 p) + hv bx (2 p) + 2 hv b^2)/2;
  g = 4 hv qRoot^2 zz^5 emc/zv;
  v11 = (lambda - 2) ar2 + hv zz^2 s1x + s1^2;
  v22 = lambda ar2 + g + hv zz^2 s2x + s2^2;
  v12 = 2 Sqrt[lambda - 2] hv qRoot zz^4 emc/Sqrt[zv];
  {{v11, v12}, {v12, v22}}
];
dxdrStar[x_?NumericQ] := -interior[[4]][x] Exp[2 x];
v11f[x_?NumericQ] := potential[x][[1, 1]];
v12f[x_?NumericQ] := potential[x][[1, 2]];
v22f[x_?NumericQ] := potential[x][[2, 2]];

encode[value_] := ToString[InputForm[N[value, 35]]];
encodeComplex[value_] := <|"re" -> encode[Re[value]], "im" -> encode[Im[value]]|>;

ClearAll[transferAt];
transferAt[omega_] := Module[
  {master, psi, mom, pK, bK, cK, hK, zK, hxK, nK, dLogAdx,
   kt, ks, kNorm, softFactor, softRow, initialPsi, initialPi,
   initialFlux, finalFlux, fluxError},
  master = NDSolveValue[
    {
      vg1'[t] == Exp[t] pg1[t]/dxdrStar[Exp[t]],
      ve1'[t] == Exp[t] pe1[t]/dxdrStar[Exp[t]],
      pg1'[t] == Exp[t] ((v11f[Exp[t]] - omega^2) vg1[t] +
          v12f[Exp[t]] ve1[t])/dxdrStar[Exp[t]],
      pe1'[t] == Exp[t] (v12f[Exp[t]] vg1[t] +
          (v22f[Exp[t]] - omega^2) ve1[t])/dxdrStar[Exp[t]],
      vg2'[t] == Exp[t] pg2[t]/dxdrStar[Exp[t]],
      ve2'[t] == Exp[t] pe2[t]/dxdrStar[Exp[t]],
      pg2'[t] == Exp[t] ((v11f[Exp[t]] - omega^2) vg2[t] +
          v12f[Exp[t]] ve2[t])/dxdrStar[Exp[t]],
      pe2'[t] == Exp[t] (v12f[Exp[t]] vg2[t] +
          (v22f[Exp[t]] - omega^2) ve2[t])/dxdrStar[Exp[t]],
      vg1[tTransferStart] == 1/Sqrt[omega], ve1[tTransferStart] == 0,
      pg1[tTransferStart] == -I Sqrt[omega], pe1[tTransferStart] == 0,
      vg2[tTransferStart] == 0, ve2[tTransferStart] == 1/Sqrt[omega],
      pg2[tTransferStart] == 0, pe2[tTransferStart] == -I Sqrt[omega]
    },
    {vg1, ve1, pg1, pe1, vg2, ve2, pg2, pe2},
    {t, tTransferStart, tTransferStop},
    WorkingPrecision -> 50, AccuracyGoal -> 18, PrecisionGoal -> 18,
    MaxSteps -> Infinity, MaxStepSize -> SetPrecision[5*10^-3, 50],
    Method -> {"ExplicitRungeKutta", "DifferenceOrder" -> 8}
  ];
  psi = {{master[[1]][tTransferStop], master[[5]][tTransferStop]},
         {master[[2]][tTransferStop], master[[6]][tTransferStop]}};
  mom = {{master[[3]][tTransferStop], master[[7]][tTransferStop]},
         {master[[4]][tTransferStop], master[[8]][tTransferStop]}};
  pK = interior[[1]][xStop]; bK = interior[[2]][xStop];
  cK = interior[[3]][xStop]; hK = interior[[4]][xStop]; zK = Exp[xStop];
  hxK = Exp[-cK] (-1/zK + qRoot^2 zK/zfun[pK]);
  nK = hK zK Exp[cK];
  dLogAdx = hxK/hK + 1 - bK^2;
  kt = -Sqrt[-nK] zK dLogAdx/2; ks = Sqrt[-nK] zK;
  kNorm = Sqrt[kt^2 + 2 ks^2];
  softFactor = omega Sqrt[lambda] Exp[cK] zK/
    (Sqrt[-nK] Sqrt[8 (lambda - 2)] kNorm);
  softRow = -I softFactor psi[[2]];
  initialPsi = IdentityMatrix[2]/Sqrt[omega];
  initialPi = -I Sqrt[omega] IdentityMatrix[2];
  initialFlux = (ConjugateTranspose[initialPsi].initialPi -
    ConjugateTranspose[initialPi].initialPsi)/(2 I);
  finalFlux = (ConjugateTranspose[psi].mom -
    ConjugateTranspose[mom].psi)/(2 I);
  fluxError = Max[Abs[Flatten[finalFlux - initialFlux]]];
  <|
    "omega_M" -> N[omega, 16],
    "soft_magnetic_row" -> (encodeComplex /@ softRow),
    "soft_magnetic_norm" -> encode[Norm[softRow]],
    "flux_residual" -> encode[fluxError]
  |>
];



z = Symbol["z"];
pin = Symbol["pin"];
profileP = -pin Tanh[pin z];
profileO = pin^2 Sech[pin z]^2/2;
profileChi = -Log[Cosh[pin z]];
profileResiduals = FullSimplify[{
  D[profileP, z] + 2 profileO,
  D[profileChi, z] - profileP,
  D[Log[profileO], z] - 2 profileP,
  profileP^2 + 2 profileO - pin^2
}, Assumptions -> {pin > 0, Element[z, Reals]}];

transferRecords = transferAt /@ frequencies;
number[text_] := ToExpression[StringReplace[
  StringReplace[text, RegularExpression["`[^*]*"] -> ""],
  "*^" -> "*10^"
]];
payload = <|
  "schema_version" -> 1,
  "threshold" -> <|
    "horizon_charge" -> encode[qHCritical],
    "critical_charge_to_mass" -> encode[qCritical],
    "legendre_residual" -> encode[Re[LegendreP[nuLegendre, (1 + aCritical)/(1 - aCritical)]]]
  |>,
  "background" -> <|
    "target_delta" -> encode[target],
    "measured_delta" -> encode[measured],
    "phi_h" -> encode[pRoot],
    "charge" -> encode[qRoot],
    "mass" -> encode[mInfinity],
    "sigma_k" -> encode[xStop],
    "phi_k" -> encode[interior[[1]][xStop]],
    "beta_k" -> encode[interior[[2]][xStop]]
  |>,
  "transfer" -> transferRecords,
  "inner_profile_residuals" -> (ToString[InputForm[#]] & /@ profileResiduals),
  "checks" -> <|
    "threshold_residual_below_1e30" -> TrueQ[Abs[Re[LegendreP[nuLegendre, (1 + aCritical)/(1 - aCritical)]]] < 10^-30],
    "background_above_threshold" -> TrueQ[measured > 0],
    "three_nonzero_transfer_rows" -> TrueQ[Length[transferRecords] == 3 && Min[number[#1["soft_magnetic_norm"]] & /@ transferRecords] > 0],
    "transfer_flux" -> TrueQ[Max[number[#1["flux_residual"]] & /@ transferRecords] < 10^-8],
    "analytic_inner_profile" -> TrueQ[profileResiduals == {0, 0, 0, 0}]
  |>
|>;
Export[FileNameJoin[{outDir, "crosscheck.json"}], payload, "RawJSON"];
If[And @@ Values[payload["checks"]], Exit[0], Exit[1]];
