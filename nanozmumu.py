"""
Example analysis module: make a dimuon mass plot from a NanoAOD
"""
import logging
import os.path

import bamboo.plots
from bamboo.analysismodules import (DataDrivenBackgroundHistogramsModule,
                                    NanoAODHistoModule, NanoAODModule,
                                    NanoAODSkimmerModule)

logger = logging.getLogger(__name__)


class Plot(bamboo.plots.Plot):
    def produceResults(self, bareResults, fbe, key=None):
        if any("__qcdScale" in h.GetName() for h in bareResults):
            hNom = next(h for h in bareResults if "__" not in h.GetName())
            prefix = f"{hNom.GetName()}__qcdScale"
            hVar_qcdScale = [h for h in bareResults if h.GetName().startswith(prefix)]
            if not all(hv.GetNcells() == hNom.GetNcells() for hv in hVar_qcdScale):
                logger.error("Variation histograms do not have the same binning as the nominal histogram")
            elif len(hVar_qcdScale) < 2:
                logger.error("At least two variations histograms must be provided")
            else:  # make an envelope from maximum deviations
                import numpy as np
                vars_cont = np.array([[hv.GetBinContent(i) for i in range(hv.GetNcells())]
                                      for hv in hVar_qcdScale])
                hVar_up = hNom.Clone(f"{prefix}up")
                hVar_down = hNom.Clone(f"{prefix}down")
                from itertools import count
                for i, vl, vh in zip(count(), np.amin(vars_cont, axis=0), np.amax(vars_cont, axis=0)):
                    hVar_down.SetBinContent(i, vl)
                    hVar_up.SetBinContent(i, vh)
                return bareResults + [hVar_up, hVar_down]
        return bareResults


class NanoZMuMuBase(NanoAODModule):
    """ Base module for NanoAOD Z->MuMu example """
    def addArgs(self, parser):
        super().addArgs(parser)
        parser.add_argument("--backend", type=str, default="dataframe",
                            help="Backend to use, 'dataframe' (default), 'lazy', or 'compiled'")
        parser.add_argument("--postprocessed", action="store_true",
                            help="Run on postprocessed NanoAOD")

    def prepareTree(self, tree, sample=None, sampleCfg=None, backend=None):
        if self.args.postprocessed:
            return self.prepare_postprocessed(tree, sample=sample, sampleCfg=sampleCfg, backend=backend)
        else:
            return self.prepare_ondemand(tree, sample=sample, sampleCfg=sampleCfg, backend=backend)

    def prepare_ondemand(self, tree, sample=None, sampleCfg=None, backend=None):
        era = sampleCfg.get("era") if sampleCfg else None
        isMC = self.isMC(sample)
        metName = "METFixEE2017" if era == "2017" else "MET"
        isNotWorker = True  # for tests - more realistic: (self.args.distributed != "worker")
        # Decorate the tree
        from bamboo.treedecorators import (NanoAODDescription, CalcCollectionsGroups,
                                           nanoRochesterCalc, nanoFatJetCalc)
        nanoJetMETCalc_both = CalcCollectionsGroups(
            Jet=("pt", "mass"), changes={metName: (f"{metName}T1", f"{metName}T1Smear")},
            **{metName: ("pt", "phi")})
        nanoJetMETCalc_data = CalcCollectionsGroups(
            Jet=("pt", "mass"), changes={metName: (f"{metName}T1",)},
            **{metName: ("pt", "phi")})
        systVars = (([nanoRochesterCalc] if era == "2016" else [])
                    + [(nanoJetMETCalc_both if isMC else nanoJetMETCalc_data), nanoFatJetCalc])
        tree, noSel, be, lumiArgs = super().prepareTree(
            tree, sample=sample, sampleCfg=sampleCfg,
            description=NanoAODDescription.get(
                "v7", year=(era if era else "2016"), isMC=isMC, systVariations=systVars),
            backend=self.args.backend or backend)
        # per-year/era options
        puWeightsFile = None
        jecTag, smearTag = None, None
        rochesterFile = None
        if era == "2016":
            rochesterFile = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "tests", "data", "RoccoR2016.txt")
            if isMC:
                puWeightsFile = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "tests", "data", "puweights.json")
                jecTag = "Summer16_07Aug2017_V11_MC"
                smearTag = "Summer16_25nsV1_MC"
            else:
                if "2016B" in sample or "2016C" in sample or "2016D" in sample:
                    jecTag = "Summer16_07Aug2017BCD_V11_DATA"
                elif "2016E" in sample or "2016F" in sample:
                    jecTag = "Summer16_07Aug2017EF_V11_DATA"
                elif "2016G" in sample or "2016H" in sample:
                    jecTag = "Summer16_07Aug2017GH_V11_DATA"
                else:
                    raise ValueError(f"Could not deduce data JEC tag for sample {sample}")
        elif era == "2017":
            if isMC:
                jecTag = "Fall17_17Nov2017_V32_MC"
                smearTag = "Fall17_V3_MC"
            else:
                if "2017B" in sample:
                    jecTag = "Fall17_17Nov2017B_V32_DATA"
                elif "2017C" in sample:
                    jecTag = "Fall17_17Nov2017C_V32_DATA"
                elif "2017D" in sample or "2017E" in sample:
                    jecTag = "Fall17_17Nov2017DE_V32_DATA"
                elif "2017F" in sample:
                    jecTag = "Fall17_17Nov2017F_V32_DATA"
                else:
                    raise ValueError(f"Could not deduce data JEC tag for sample {sample}")
        # always-on event weights
        if isMC:
            mcWgts = [tree.genWeight]
            if puWeightsFile:
                from bamboo.analysisutils import makePileupWeight
                mcWgts.append(makePileupWeight(
                    puWeightsFile, tree.Pileup_nTrueInt,
                    nameHint="bamboo_puWeight{}".format("".join(c for c in sample if c.isalnum()))))
                # mcWgts.append(makePileupWeight((
                #     "/afs/cern.ch/user/p/piedavid/private/bamboodev/testUL.json",
                #     "Collisions16_UltraLegacy_goldenJSON"), tree.Pileup_nTrueInt, systName="pu", sel=noSel))
            else:
                logger.warning("Running on MC without pileup reweighting")
            from bamboo import treefunctions as op
            mcWgts += [
                op.systematic(op.c_float(1.), **{f"qcdScale{i:d}": tree.LHEScaleWeight[i]
                                                 for i in (0, 1, 3, 5, 7, 8)}),
                # should only be added when present
                # op.systematic(op.c_float(1.), name="psISR", up=tree.PSWeight[2], down=tree.PSWeight[0]),
                # op.systematic(op.c_float(1.), name="psFSR", up=tree.PSWeight[3], down=tree.PSWeight[1]),
            ]
            noSel = noSel.refine("mcWeight", weight=mcWgts)

        cmJMEArgs = {
            "jec": jecTag,
            "smear": smearTag,
            "splitJER": True,
            "jesUncertaintySources": ("All" if isMC else None),
            # Alternative, for regrouped
            # "jesUncertaintySources": ("Merged" if isMC else None),
            # "regroupTag": "V2",
            "mayWriteCache": isNotWorker,
            "isMC": isMC,
            "backend": be,
        }
        # configure corrections and variations
        from bamboo.analysisutils import configureJets, configureType1MET, configureRochesterCorrection
        configureJets(tree._Jet, "AK4PFchs", **cmJMEArgs)
        if isMC:
            configureType1MET(getattr(tree, f"_{metName}T1Smear"), isT1Smear=True, **cmJMEArgs)
        configureType1MET(
            getattr(tree, f"_{metName}T1"),
            enableSystematics=((lambda v: not v.startswith("jer")) if isMC else None),
            **cmJMEArgs)
        cmJMEArgs.update({"jesUncertaintySources": (["Total"] if isMC else None), "regroupTag": None})
        configureJets(tree._FatJet, "AK8PFPuppi", mcYearForFatJets=era, **cmJMEArgs)
        if era == "2016":
            configureRochesterCorrection(tree._Muon, rochesterFile, isMC=isMC, backend=be)

        return tree, noSel, be, lumiArgs

    def prepare_postprocessed(self, tree, sample=None, sampleCfg=None):
        era = sampleCfg.get("era") if sampleCfg else None
        isMC = self.isMC(sample)
        metName = "METFixEE2017" if era == "2017" else "MET"
        # Decorate the tree
        from bamboo.treedecorators import NanoAODDescription, ReadJetMETVar, NanoReadRochesterVar
        nanoReadJetMETVar_MC = ReadJetMETVar(
            "Jet", f"{metName}_T1Smear",
            bTaggers=["csvv2", "deepcsv", "deepjet", "cmva"],
            bTagWPs=["L", "M", "T", "shape"])
        nanoReadNosmearMETVar = ReadJetMETVar(None, f"{metName}_T1")
        systVars = [nanoReadNosmearMETVar]
        if era == "2016":
            systVars.append(NanoReadRochesterVar())
        if isMC:
            systVars += [nanoReadJetMETVar_MC]
        tree, noSel, be, lumiArgs = super().prepareTree(
            tree, sample=sample, sampleCfg=sampleCfg,
            description=NanoAODDescription.get(
                "v7", year=(era if era else "2016"), isMC=isMC,
                removeGroups=f"{metName}_",
                addGroups=[f"{metName}_T1_"] + ([f"{metName}_T1Smear_"] if isMC else []),
                systVariations=systVars))

        # always-on event weights
        if isMC:
            mcWgts = [tree.genWeight]
            if era == "2016":
                mcWgts.append(tree.puWeight)
            from bamboo import treefunctions as op
            mcWgts += [
                op.systematic(
                    op.c_float(1.), name="qcdScale",
                    **{f"qcdScale{i:d}": tree.LHEScaleWeight[i] for i in (0, 1, 3, 5, 7, 8)}),
                # should only be added when present
                # op.systematic(op.c_float(1.), name="psISR", up=tree.PSWeight[2], down=tree.PSWeight[0]),
                # op.systematic(op.c_float(1.), name="psFSR", up=tree.PSWeight[3], down=tree.PSWeight[1]),
            ]
            noSel = noSel.refine("mcWeight", weight=mcWgts)

        return tree, noSel, be, lumiArgs


class NanoZMuMu(NanoZMuMuBase, DataDrivenBackgroundHistogramsModule):
    """ Example module: Z->MuMu histograms from NanoAOD """
    def definePlots(self, t, noSel, sample=None, sampleCfg=None):
        from bamboo.plots import CutFlowReport, SummedPlot, Skim
        from bamboo.plots import EquidistantBinning as EqB
        from bamboo import treefunctions as op
        from bamboo.analysisutils import forceDefine

        era = sampleCfg.get("era") if sampleCfg else None

        plots = []
        cfr = CutFlowReport("yields", recursive=True)
        plots.append(cfr)

        muons = op.select(t.Muon, lambda mu: op.AND(mu.pt > 20., op.abs(mu.eta) < 2.4))
        electrons = op.select(t.Electron, lambda el: op.AND(el.pt > 20, op.abs(el.eta) < 2.5))

        twoMuSel = noSel.refine("twoMuons", cut=[op.rng_len(muons) > 1])
        cfr.add(twoMuSel, "With two muons")
        cfr.add(twoMuSel, "With two leptons")
        plots.append(Plot.make1D(
            "dimu_M", op.invariant_mass(muons[0].p4, muons[1].p4), twoMuSel, EqB(100, 20., 120.),
            title="Dimuon invariant mass", plotopts={"show-overflow": False}))
        if self.args.backend != "compiled":
            plots.append(Skim(
                "muSkim", {
                    "nSelMuons": op.static_cast("UInt_t", op.rng_len(muons)),  # TBranch doesn't accept size_t
                    "selMuons_i": muons.idxs,
                    "selMu_miniPFRelIsoNeu": op.map(
                        muons, lambda mu: mu.miniPFRelIso_all - mu.miniPFRelIso_chg)
                }, twoMuSel,
                keepOriginal=[Skim.KeepRegex("PV_.*"), "nOtherPV", Skim.KeepRegex("OtherPV_.*")]))

        # evaluate jet and MET for all events passing twoMuSel
        # more optimization will be needed with systematics etc.
        metName = "METFixEE2017" if era == "2017" else "MET"
        if not self.args.postprocessed:
            for calcProd in t._Jet.calcProds:
                forceDefine(calcProd, twoMuSel)
            for calcProd in getattr(t, f"_{metName}T1").calcProds:
                forceDefine(calcProd, twoMuSel)
            if self.isMC(sample):
                for calcProd in getattr(t, f"_{metName}T1Smear").calcProds:
                    forceDefine(calcProd, twoMuSel)
        else:
            metName += "_"

        jets_noclean = op.select(t.Jet, lambda j: op.AND(j.jetId & 0x2, op.abs(j.eta) < 2.4, j.pt > 20.))
        jets = op.sort(
            op.select(jets_noclean, lambda j: op.AND(
                op.NOT(op.rng_any(muons, lambda l: op.deltaR(l.p4, j.p4) < 0.4)),
                op.NOT(op.rng_any(electrons, lambda l: op.deltaR(l.p4, j.p4) < 0.4))
            )), lambda j: -j.pt)

        plots.append(Plot.make1D(
            "nJets", op.rng_len(jets), twoMuSel, EqB(10, 0., 10.), title="Number of jets"))

        twoMuTwoJetSel = twoMuSel.refine("twoMuonsTwoJets", cut=[op.rng_len(jets) > 1])
        cfr.add(twoMuTwoJetSel, "With two muons and two jets")

        leadjpt = Plot.make1D(
            "leadJetPT", jets[0].pt, twoMuTwoJetSel, EqB(50, 0., 250.), title="Leading jet PT")
        # leadjphi = Plot.make1D(
        #     "leadJetPHI", op.Phi_mpi_pi(jets[0].phi), twoMuTwoJetSel, EqB(50, -3.142, 3.142),
        #     title="Leading jet PHI")
        subleadjpt = Plot.make1D(
            "subleadJetPT", jets[1].pt, twoMuTwoJetSel, EqB(50, 0., 250.), title="Subleading jet PT")
        plots += [leadjpt, subleadjpt]
        plots.append(SummedPlot(
            "twoLeadJetPT", [leadjpt, subleadjpt], xTitle="Leading two jet PTs"))
        metT1 = getattr(t, f"{metName}T1")
        if self.isMC(sample):
            metT1Smear = getattr(t, f"{metName}T1Smear")
        else:
            metT1Smear = metT1
        plots.append(Plot.make1D(
            "METT1", metT1.pt, twoMuTwoJetSel, EqB(50, 0., 250.), title="METT1 PT"))
        plots.append(Plot.make1D(
            "METT1Smear", metT1Smear.pt, twoMuTwoJetSel, EqB(50, 0., 250.), title="METT1Smear PT"))

        deepCSVFile = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "tests", "data", "DeepCSV_2016LegacySF_V1.csv")
        if os.path.exists(deepCSVFile):  # protection for CI tests
            from bamboo.scalefactors import BtagSF
            sf_deepcsv = BtagSF("csv", deepCSVFile, wp="Loose", sysType="central",
                                otherSysTypes=("up", "down"), measurementType="comb",
                                sel=noSel, uName=sample)
            # for reshaping: add getters={"Discri": lambda j : j.btagDeepB}

            bJets_DeepCSVLoose = op.select(jets, lambda j: j.btagDeepB > 0.2217)
            bTagSel = twoMuTwoJetSel.refine(
                "twoMuonsTwoJetsB", cut=[op.rng_len(bJets_DeepCSVLoose) > 0],
                weight=(sf_deepcsv(bJets_DeepCSVLoose[0]) if self.isMC(sample) else None))
            cfr.add(bTagSel, "With two muons a b-tag")
            plots.append(Plot.make1D(
                "bjetpt", bJets_DeepCSVLoose[0].pt, bTagSel, EqB(50, 0., 250.), title="B-jet pt"))

        from bamboo.scalefactors import get_scalefactor, binningVariables_nano
        elIDSF = get_scalefactor(
            "lepton",
            os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "tests", "data", "Electron_EGamma_SF2D_loose_moriond17.json"),
            isElectron=True,
            paramDefs=binningVariables_nano)
        if self.isMC(sample):
            genHardElectrons = op.select(t.GenPart, lambda gp: op.AND(
                (gp.statusFlags & (0x1 << 7)),
                op.abs(gp.pdgId) == 11))
            plots.append(Plot.make1D(
                "nGenElectrons", op.rng_len(genHardElectrons), noSel, EqB(5, 0., 5.),
                title="Number of electrons/positrons in the matrix element"))
            from bamboo.plots import LateSplittingSelection
            noSel = LateSplittingSelection.create(
                noSel, "splitByGenEl", keepInclusive=True, splitCuts={
                    "2El": op.rng_len(genHardElectrons) == 2,
                    "no2El": op.rng_len(genHardElectrons) != 2
                })
        twoElSel = noSel.refine(
            "twoElectrons", cut=[op.rng_len(electrons) > 1],
            weight=[elIDSF(electrons[i]) for i in range(2)])
        cfr.add(twoElSel, "With two leptons")
        plots.append(Plot.make1D(
            "Melel", op.invariant_mass(electrons[0].p4, electrons[1].p4), twoElSel,
            EqB(100, 20., 120.), title="Dielectron invariant mass",
            plotopts={"show-overflow": False}))

        return plots


class SkimNanoZMuMu(NanoZMuMuBase, NanoAODSkimmerModule):
    def defineSkimSelection(self, tree, noSel, sample=None, sampleCfg=None):
        from bamboo import treefunctions as op
        muons = op.select(tree.Muon, lambda mu: op.AND(mu.pt > 20., op.abs(mu.eta) < 2.4))
        hasTwoMu = noSel.refine("hasTwoMu", cut=(op.rng_len(muons) >= 2))
        varsToKeep = {"nMuon": None, "Muon_eta": None, "Muon_pt": None}  # from input file
        varsToKeep["nSelMuons"] = op.static_cast("UInt_t", op.rng_len(muons))  # TBranch doesn't accept size_t
        varsToKeep["selMuons_i"] = muons.idxs
        varsToKeep["selMu_miniPFRelIsoNeu"] = op.map(
            muons, lambda mu: mu.miniPFRelIso_all - mu.miniPFRelIso_chg)
        return hasTwoMu, varsToKeep


class SkimNanoAOD(NanoAODSkimmerModule):
    def prepareTree(self, tree, sample=None, sampleCfg=None, backend=None):
        era = sampleCfg.get("era") if sampleCfg else None
        from bamboo.treedecorators import NanoAODDescription
        return super(NanoAODSkimmerModule, self).prepareTree(
            tree, sample=sample, sampleCfg=sampleCfg,
            description=NanoAODDescription.get("v7", year=(era if era else "2016")))


class MinimalNanoZMuMu(NanoAODHistoModule):
    def addArgs(self, parser):
        super().addArgs(parser)
        parser.add_argument(
            "--backend", type=str, default="dataframe",
            help="Backend to use, 'dataframe' (default), 'lazy', or 'compiled'")

    def prepareTree(self, tree, sample=None, sampleCfg=None, backend=None):
        from bamboo.treedecorators import NanoAODDescription
        return super().prepareTree(
            tree, sample=sample, sampleCfg=sampleCfg,
            description=NanoAODDescription.get(
                "v5", year="2016", isMC=self.isMC(sample)), backend=self.args.backend)

    def definePlots(self, t, noSel, sample=None, sampleCfg=None):
        from bamboo.plots import Plot
        from bamboo.plots import EquidistantBinning as EqB
        from bamboo import treefunctions as op
        if self.isMC(sample):
            noSel = noSel.refine("mcWeight", weight=t.genWeight)
        plots = []
        muons = op.select(
            t.Muon, lambda mu: op.AND(
                mu.mediumId,
                mu.pfRelIso03_all < 0.4,
                mu.pt > 15.))
        twoMuSel = noSel.refine("has2mu", cut=(op.rng_len(muons) > 1))
        plots.append(Plot.make1D(
            "dimuM", (muons[0].p4 + muons[1].p4).M(), twoMuSel,
            EqB(100, 20., 120.), title="Invariant mass"))
        return plots
