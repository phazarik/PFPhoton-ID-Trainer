//
// Original Author:  Ashim Roy
//         Created:  Sat, 13 Jun 2020 20:37:58 GMT

#include "../interface/egmNtuplizer.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include <cmath>

using namespace std;
using namespace edm;

Int_t          nPho_;
vector<float>  phoE_;
vector<float>  phoEt_;
vector<float>  phoPt_;
vector<float>  phoEta_;
vector<float>  phoPhi_;
vector<float>  phoSigmaE_;
vector<float>  phoCalibE_;
vector<float>  phoCalibEt_;
vector<float>  phoSCE_;
vector<float>  phoSCRawE_;
vector<float>  phoESEn_;
vector<float>  phoESEnP1_;
vector<float>  phoESEnP2_;
vector<float>  phoSCEta_;
vector<float>  phoSCPhi_;
vector<float>  phoSCEtaWidth_;
vector<float>  phoSCPhiWidth_;
vector<float>  phoSCBrem_;
vector<int>    phohasPixelSeed_;
vector<int>    phoEleVeto_;
vector<float>  phoR9_;
vector<float>  phoHoverE_;
vector<float>  phoESEffSigmaRR_;
vector<float>  phoSigmaIEtaIEtaFull5x5_;
vector<float>  phoSigmaIEtaIPhiFull5x5_;
vector<float>  phoSigmaIPhiIPhiFull5x5_;
vector<float>  phoE2x2Full5x5_;
vector<float>  phoE5x5Full5x5_;
vector<float>  phoR9Full5x5_;
vector<float>  phoPFChIso_;
vector<float>  phoPFChPVIso_;
vector<float>  phoPFPhoIso_;
vector<float>  phoPFNeuIso_;
vector<float>  phoPFChWorstVetoIso_;
vector<float>  phoPFChWorstIso_;
vector<float>  phoEcalPFClusterIso_;
vector<float>  phoHcalPFClusterIso_;
//vector<float>  phoSeedBCE_;
//vector<float>  phoSeedBCEta_;
vector<float>  phoIDMVA_;
vector<ULong64_t> phoFiredSingleTrgs_;
vector<ULong64_t> phoFiredDoubleTrgs_;
vector<ULong64_t> phoFiredTripleTrgs_;
vector<ULong64_t> phoFiredL1Trgs_;
vector<float>  phoSeedTime_;
vector<float>  phoSeedEnergy_;
vector<int>  phoSeediEta_;
vector<int>  phoSeediPhi_;
vector<vector<float>> phoEnergyMatrix5x5_;
vector<vector<float>> phoEnergyMatrix7x7_;
vector<vector<float>> phoEnergyMatrix9x9_;
vector<vector<float>> phoEnergyMatrix11x11_;
vector<vector<float>> phoEnergyMatrix15x15_;

//New:
vector<float> hadTowOverEmValid_;
vector<float> trkSumPtHollow_;
vector<float> trkSumPtSolid_;
vector<float> ecalRecHit_;
vector<float> hcalTower_;
vector<float> sigmaIetaIeta_;
vector<float> hadOverEmCone_;

//Flags:
vector<int> isPhotonMatching_;
vector<int> isPromptFinalState_;
vector<int> isDirectHardProcessTauDecayProductFinalState_;
vector<int> isDirectHadronDecayProduct_;

vector<int> isTauDecayProduct_;
vector<int> isPromptTauDecayProduct_;
vector<int> isDirectTauDecayProduct_;
vector<int> isDirectPromptTauDecayProduct_;
vector<int> isDirectPromptTauDecayProductFinalState_;
vector<int> isHardProcess_;
vector<int> fromHardProcessFinalState_;
vector<int> fromHardProcessDecayed_;
vector<float> phoMatchingGenPt_;
vector<int> isPionMother_;
vector<int> isPFPhoton_;

//Necessary for the Photon Footprint removal
template <class T, class U>
bool isInFootprint(const T& thefootprint, const U& theCandidate) {
  for ( auto itr = thefootprint.begin(); itr != thefootprint.end(); ++itr ) {

    if( itr.key() == theCandidate.key() ) return true;
    
  }
  return false;
}

void egmNtuplizer::branchesPhotons(TTree* tree) {

  //////////////////////////////Photon Branches/////////////////////////////////////////
  tree->Branch("nPho",                    &nPho_);
  tree->Branch("phoE",                    &phoE_);
  tree->Branch("phoEt",                   &phoEt_);
  tree->Branch("phoPt",                   &phoPt_);
  tree->Branch("phoEta",                  &phoEta_);
  tree->Branch("phoPhi",                  &phoPhi_);
  //tree->Branch("phoSigmaE",               &phoSigmaE_);
  //tree->Branch("phoCalibE",               &phoCalibE_);
  //tree->Branch("phoCalibEt",              &phoCalibEt_);
  tree->Branch("phoSCE",                  &phoSCE_);
  tree->Branch("phoSCRawE",               &phoSCRawE_);
  tree->Branch("phoESEn",                 &phoESEn_);
  tree->Branch("phoESEnP1",               &phoESEnP1_);
  tree->Branch("phoESEnP2",               &phoESEnP2_);
  tree->Branch("phoSCEta",                &phoSCEta_);
  tree->Branch("phoSCPhi",                &phoSCPhi_);
  tree->Branch("phoSCEtaWidth",           &phoSCEtaWidth_);
  tree->Branch("phoSCPhiWidth",           &phoSCPhiWidth_);
  tree->Branch("phoSCBrem",               &phoSCBrem_);
  tree->Branch("phohasPixelSeed",         &phohasPixelSeed_);
  tree->Branch("phoEleVeto",              &phoEleVeto_);
  tree->Branch("phoR9",                   &phoR9_);
  tree->Branch("phoHoverE",               &phoHoverE_);
  tree->Branch("phoESEffSigmaRR",         &phoESEffSigmaRR_);
  tree->Branch("phoSigmaIEtaIEtaFull5x5", &phoSigmaIEtaIEtaFull5x5_);
  tree->Branch("phoSigmaIEtaIPhiFull5x5", &phoSigmaIEtaIPhiFull5x5_);
  tree->Branch("phoSigmaIPhiIPhiFull5x5", &phoSigmaIPhiIPhiFull5x5_);
  tree->Branch("phoE2x2Full5x5",          &phoE2x2Full5x5_);
  tree->Branch("phoE5x5Full5x5",          &phoE5x5Full5x5_);
  tree->Branch("phoR9Full5x5",            &phoR9Full5x5_);
  tree->Branch("phoPFChIso",              &phoPFChIso_);
  tree->Branch("phoPFChPVIso",            &phoPFChPVIso_);
  tree->Branch("phoPFPhoIso",             &phoPFPhoIso_);
  tree->Branch("phoPFNeuIso",             &phoPFNeuIso_);
  tree->Branch("phoPFChWorstIso",         &phoPFChWorstIso_);
  tree->Branch("phoPFChWorstVetoIso",     &phoPFChWorstVetoIso_);
  tree->Branch("phoEcalPFClusterIso",     &phoEcalPFClusterIso_);
  tree->Branch("phoHcalPFClusterIso",     &phoHcalPFClusterIso_);
  tree->Branch("phoSeedTime",             &phoSeedTime_);
  tree->Branch("phoSeedEnergy",           &phoSeedEnergy_);
  tree->Branch("phoSeediEta",             &phoSeediEta_);
  tree->Branch("phoSeediPhi",             &phoSeediPhi_);
  tree->Branch("phoEnergyMatrix5x5",      &phoEnergyMatrix5x5_);
  tree->Branch("phoEnergyMatrix7x7",      &phoEnergyMatrix7x7_);
  tree->Branch("phoEnergyMatrix9x9",      &phoEnergyMatrix9x9_);
  tree->Branch("phoEnergyMatrix11x11",    &phoEnergyMatrix11x11_);
  tree->Branch("phoEnergyMatrix15x15",    &phoEnergyMatrix15x15_);

  //New :
  tree->Branch("phohadOverEmCone",        &hadOverEmCone_);
  tree->Branch("phohadTowOverEmValid",    &hadTowOverEmValid_);
  tree->Branch("photrkSumPtHollow",       &trkSumPtHollow_);
  tree->Branch("photrkSumPtSolid",        &trkSumPtSolid_);
  tree->Branch("phoecalRecHit",           &ecalRecHit_);
  tree->Branch("phohcalTower",            &hcalTower_);
  tree->Branch("phosigmaIetaIeta",        &sigmaIetaIeta_);
  
  tree->Branch("isPhotonMatching",        &isPhotonMatching_);
  tree->Branch("isPromptFinalState",      &isPromptFinalState_);
  tree->Branch("isDirectHardProcessTauDecayProductFinalState", &isDirectHardProcessTauDecayProductFinalState_);
  tree->Branch("isDirectHadronDecayProduct", &isDirectHadronDecayProduct_);
  tree->Branch("isTauDecayProduct",       &isTauDecayProduct_);
  tree->Branch("isPromptTauDecayProduct", &isPromptTauDecayProduct_);
  tree->Branch("isDirectTauDecayProduct", &isDirectTauDecayProduct_);
  tree->Branch("isDirectPromptTauDecayProduct", &isDirectPromptTauDecayProduct_);
  tree->Branch("isDirectPromptTauDecayProductFinalState", &isDirectPromptTauDecayProductFinalState_);
  tree->Branch("isHardProcess",           &isHardProcess_);
  tree->Branch("fromHardProcessFinalState", &fromHardProcessFinalState_);
  tree->Branch("fromHardProcessDecayed",  &fromHardProcessDecayed_);
  tree->Branch("phoMatchingGenPt",        &phoMatchingGenPt_);
  tree->Branch("isPionMother",            &isPionMother_);
  tree->Branch("isPFPhoton",              &isPFPhoton_);
}

//
// member functions
//
// ------------ method called for each event  ------------
void egmNtuplizer::fillPhotons(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  //////////////////////////////Cleaning of previous event/////////////////////////////////////////
  phoE_                   .clear();
  phoSigmaE_              .clear();
  phoEt_                  .clear();
  phoPt_                  .clear();
  phoEta_                 .clear();
  phoPhi_                 .clear();
  phoCalibE_              .clear();
  phoCalibEt_             .clear();
  phoSCE_                 .clear();
  phoSCRawE_              .clear();
  phoESEn_                .clear();
  phoESEnP1_              .clear();
  phoESEnP2_              .clear();
  phoSCEta_               .clear();
  phoSCPhi_               .clear();
  phoSCEtaWidth_          .clear();
  phoSCPhiWidth_          .clear();
  phoSCBrem_              .clear();
  phohasPixelSeed_        .clear();
  phoEleVeto_             .clear();
  phoR9_                  .clear();
  phoHoverE_              .clear();
  phoESEffSigmaRR_        .clear();
  phoSigmaIEtaIEtaFull5x5_.clear();
  phoSigmaIEtaIPhiFull5x5_.clear();
  phoSigmaIPhiIPhiFull5x5_.clear();
  phoE2x2Full5x5_         .clear();
  phoE5x5Full5x5_         .clear();
  phoR9Full5x5_           .clear();
  phoPFChIso_             .clear();
  phoPFChPVIso_           .clear();
  phoPFPhoIso_            .clear();
  phoPFNeuIso_            .clear();
  phoPFChWorstVetoIso_    .clear();
  phoPFChWorstIso_        .clear();
  phoEcalPFClusterIso_    .clear();
  phoHcalPFClusterIso_    .clear();
  phoSeedTime_            .clear();
  phoSeedEnergy_          .clear();
  phoSeediEta_            .clear();
  phoSeediPhi_            .clear();
  phoEnergyMatrix5x5_     .clear();
  phoEnergyMatrix7x7_     .clear();
  phoEnergyMatrix9x9_     .clear();
  phoEnergyMatrix11x11_   .clear();
  phoEnergyMatrix15x15_   .clear();
  nPho_ = 0;

  //New :
  hadOverEmCone_          .clear();
  hadTowOverEmValid_      .clear();
  trkSumPtHollow_         .clear();
  trkSumPtSolid_          .clear();
  ecalRecHit_             .clear();
  hcalTower_              .clear();
  sigmaIetaIeta_          .clear();

  isPhotonMatching_       .clear();
  isPromptFinalState_     .clear();
  isDirectHardProcessTauDecayProductFinalState_ .clear();
  isDirectHadronDecayProduct_ .clear();

  isTauDecayProduct_      .clear();
  isPromptTauDecayProduct_.clear();
  isDirectTauDecayProduct_.clear();
  isDirectPromptTauDecayProduct_.clear();
  isDirectPromptTauDecayProductFinalState_.clear();
  isHardProcess_          .clear();
  fromHardProcessFinalState_.clear();
  fromHardProcessDecayed_ .clear();
  
  phoMatchingGenPt_       .clear();
  isPionMother_           .clear();
  isPFPhoton_             .clear();


  //////////////////////////////Filling of Variables///////////////////////////////////
  edm::Handle<edm::View<pat::Photon> > photonHandle;
  iEvent.getByToken(photonCollection_, photonHandle);

  if (!photonHandle.isValid()) {
    edm::LogWarning("ggNtuplizer") << "no pat::Photons in event";
    return;
  }

  EcalClusterLazyTools       lazyTool    (iEvent, iSetup, ebReducedRecHitCollection_, eeReducedRecHitCollection_, esReducedRecHitCollection_);
  noZS::EcalClusterLazyTools lazyToolnoZS(iEvent, iSetup, ebReducedRecHitCollection_, eeReducedRecHitCollection_, esReducedRecHitCollection_);

  for (edm::View<pat::Photon>::const_iterator iPho = photonHandle->begin(); iPho != photonHandle->end(); ++iPho) {
    phoE_               .push_back(iPho->energy());
    phoEt_              .push_back(iPho->et());
    phoPt_              .push_back(iPho->pt());
    ////AR Requested UserFloat ecalEnergyPostCorr is not available! Possible UserFloats are: 
    ////AR  PhotonMVAEstimatorRun2Spring16NonTrigV1Values PhotonMVAEstimatorRunIIFall17v1p1Values PhotonMVAEstimatorRunIIFall17v2Values 
    //phoCalibE_          .push_back(iPho->userFloat("ecalEnergyPostCorr"));
    //phoCalibEt_         .push_back(iPho->et()*iPho->userFloat("ecalEnergyPostCorr")/iPho->energy());
    //phoSigmaE_          .push_back(iPho->userFloat("ecalEnergyErrPostCorr"));
    phoEta_             .push_back(iPho->eta());
    phoPhi_             .push_back(iPho->phi());
    phoSCE_             .push_back((*iPho).superCluster()->energy());
    phoSCRawE_          .push_back((*iPho).superCluster()->rawEnergy());
    phoESEn_            .push_back((*iPho).superCluster()->preshowerEnergy());
    phoESEnP1_          .push_back((*iPho).superCluster()->preshowerEnergyPlane1());
    phoESEnP2_          .push_back((*iPho).superCluster()->preshowerEnergyPlane2());
    phoSCEta_           .push_back((*iPho).superCluster()->eta());
    phoSCPhi_           .push_back((*iPho).superCluster()->phi());
    phoSCEtaWidth_      .push_back((*iPho).superCluster()->etaWidth());
    phoSCPhiWidth_      .push_back((*iPho).superCluster()->phiWidth());
    phoSCBrem_          .push_back((*iPho).superCluster()->phiWidth()/(*iPho).superCluster()->etaWidth());
    phohasPixelSeed_    .push_back((Int_t)iPho->hasPixelSeed());
    phoEleVeto_         .push_back((Int_t)iPho->passElectronVeto());
    phoR9_              .push_back(iPho->r9());
    phoHoverE_          .push_back(iPho->hadTowOverEm());
    phoESEffSigmaRR_    .push_back(lazyTool.eseffsirir(*((*iPho).superCluster())));
    phoSigmaIEtaIEtaFull5x5_ .push_back(iPho->full5x5_sigmaIetaIeta());
    phoSigmaIEtaIPhiFull5x5_ .push_back(iPho->full5x5_showerShapeVariables().sigmaIetaIphi);
    phoSigmaIPhiIPhiFull5x5_ .push_back(iPho->full5x5_showerShapeVariables().sigmaIphiIphi);
    phoE2x2Full5x5_          .push_back(lazyToolnoZS.e2x2(*((*iPho).superCluster()->seed())));
    phoE5x5Full5x5_          .push_back(iPho->full5x5_e5x5());
    phoR9Full5x5_            .push_back(iPho->full5x5_r9());    
    phoPFChIso_         .push_back(iPho->chargedHadronIso()); //charged hadron isolation with dxy,dz match to pv
    phoPFChPVIso_       .push_back(iPho->chargedHadronPFPVIso()); //only considers particles assigned to the primary vertex (PV) by particle flow, corresponds to <10_6 chargedHadronIso
    phoPFPhoIso_        .push_back(iPho->photonIso());
    phoPFNeuIso_        .push_back(iPho->neutralHadronIso());
    phoPFChWorstIso_    .push_back(iPho->chargedHadronWorstVtxIso()); //max charged hadron isolation when dxy/dz matching to given vtx
    phoPFChWorstVetoIso_.push_back(iPho->chargedHadronWorstVtxGeomVetoIso()); //as chargedHadronWorstVtxIso but an additional geometry based veto cone
    phoEcalPFClusterIso_.push_back(iPho->ecalPFClusterIso());
    phoHcalPFClusterIso_.push_back(iPho->hcalPFClusterIso());

    const auto &seedSC = *(iPho->superCluster()->seed());
    DetId seedDetId = (iPho->superCluster()->seed()->hitsAndFractions())[0].first;
    bool isBarrel = seedDetId.subdetId() == EcalBarrel;
    const EcalRecHitCollection * rechits = (isBarrel?lazyTool.getEcalEBRecHitCollection():lazyTool.getEcalEERecHitCollection());
            
    EcalRecHitCollection::const_iterator theSeedHit = rechits->find(seedDetId);
    if (theSeedHit != rechits->end()) {
      //std::cout<<"(*theSeedHit).time()"<<(*theSeedHit).time()<<"seed energy: "<<(*theSeedHit).energy()<<std::endl;  

      phoSeedTime_  .push_back((*theSeedHit).time());
      phoSeedEnergy_.push_back((*theSeedHit).energy());
      if (isBarrel) {
	EBDetId det = theSeedHit->id();
	phoSeediEta_  .push_back(det.ieta());
	phoSeediPhi_  .push_back(det.iphi());
      }
      else {
	EEDetId det = theSeedHit->id();
	phoSeediEta_  .push_back(det.ix());
	phoSeediPhi_  .push_back(det.iy());
      }

    } else{
      phoSeedTime_  .push_back(-99.);
      phoSeedEnergy_.push_back(-99.);
      phoSeediEta_  .push_back(-999);
      phoSeediPhi_  .push_back(-999);
    }

    phoEnergyMatrix5x5_   .push_back(lazyToolnoZS.energyMatrix(seedSC,2));
    phoEnergyMatrix7x7_   .push_back(lazyToolnoZS.energyMatrix(seedSC,3));
    phoEnergyMatrix9x9_   .push_back(lazyToolnoZS.energyMatrix(seedSC,4));
    phoEnergyMatrix11x11_ .push_back(lazyToolnoZS.energyMatrix(seedSC,5));
    phoEnergyMatrix15x15_ .push_back(lazyToolnoZS.energyMatrix(seedSC,7));
    
    //New:
    hadOverEmCone_       .push_back(iPho->hadronicOverEm());
    hadTowOverEmValid_   .push_back(iPho->hadTowOverEmValid());
    trkSumPtHollow_      .push_back(iPho->trkSumPtHollowConeDR03());
    trkSumPtSolid_       .push_back(iPho->trkSumPtSolidConeDR03());
    ecalRecHit_          .push_back(iPho->ecalRecHitSumEtConeDR03());
    hcalTower_           .push_back(iPho->hcalTowerSumEtConeDR03());
    sigmaIetaIeta_       .push_back(iPho->sigmaIetaIeta());

    //#####################################################################
    //Adding the PF-flag :
    bool fillPFflag = true;
    //flag1
    if(iPho->pt() < 10.0){
      fillPFflag = false;
    }
    //flag2
    if(iPho->hadTowOverEm() > 0.05){
      fillPFflag = false;
    }
    //flag3
    if(iPho->trkSumPtHollowConeDR03() + iPho->ecalRecHitSumEtConeDR03() + iPho->hcalTowerSumEtConeDR03() > 10.0){
      fillPFflag = false;
    }
    //flag4
    if(iPho->hadTowOverEmValid() != 0 && iPho->trkSumPtSolidConeDR03() > 10.0 + 0.3*iPho->pt()){
      fillPFflag = false;
    }
    //flag5
    if(abs(iPho->eta()) < 1.442 && iPho->sigmaIetaIeta() > 0.0125){ //barrel
      fillPFflag = false;
    }
    if(abs(iPho->eta()) > 1.566 && iPho->sigmaIetaIeta() > 0.034){ //endcap
      fillPFflag = false;
    }

    //Filll up the flag :
    if(fillPFflag == true){
      isPFPhoton_   .push_back(1);
    }
    else{
      isPFPhoton_   .push_back(0);
    }
    //#####################################################################

    //GEN-MATCHING:
    float dR_min = 1000.0;
    edm::Handle<vector<reco::GenParticle> > genParticlesHandle;
    iEvent.getByToken(genParticlesCollection_, genParticlesHandle);
    if (!genParticlesHandle.isValid()) {
      edm::LogWarning("egmNtuplizer") << "no reco::GenParticles in event";
      return;
    }
    int genIndex = 0;
    int matching_index=-1;
    for (vector<reco::GenParticle>::const_iterator ip = genParticlesHandle->begin(); ip != genParticlesHandle->end(); ++ip) {
      genIndex++;
      if(ip->pdgId() == 22 && ip->status()==1){ //if the gen particle is a real photon,
	//calculate dR, and if it is small than dR_min, set dR_min to dR
	float deta = iPho->eta() - ip->eta();
	float dphi = iPho->phi() - ip->phi();
	if(dphi>M_PI)//if the difference is over pi, then we must take the other side of the angle
	  dphi = 2*M_PI-dphi;
	float dR_sq = deta*deta + dphi*dphi;
	float dR = sqrt(dR_sq);
	if(dR<dR_min){
	  dR_min=dR;
	  matching_index = genIndex;
	}
      }
    }
    //The value of dR_min after this loop is the lowest possible value for this particular photon.
    //We choose dR_cut at 0.1
    if(dR_min<0.1){
      isPhotonMatching_.push_back(1);
      //phoMatchingGenPt_.push_back("Pt of the genphoton (matching_index)");
      int index=0;
      for (vector<reco::GenParticle>::const_iterator ip = genParticlesHandle->begin(); ip != genParticlesHandle->end(); ++ip){
	index++;
	if(ip->pdgId() == 22 && ip->status()==1 && index==matching_index){//photon, real, matching
	  phoMatchingGenPt_                                    .push_back(ip->pt());
	  isPromptFinalState_                                  .push_back(ip->isPromptFinalState());
	  isDirectHardProcessTauDecayProductFinalState_        .push_back(ip->isDirectHardProcessTauDecayProductFinalState());
	  isDirectHadronDecayProduct_                          .push_back(ip->statusFlags().isDirectHadronDecayProduct());
	  
	  isTauDecayProduct_                                   .push_back(ip->statusFlags().isTauDecayProduct());
	  isPromptTauDecayProduct_                             .push_back(ip->statusFlags().isPromptTauDecayProduct());
	  isDirectTauDecayProduct_                             .push_back(ip->statusFlags().isDirectTauDecayProduct());
	  isDirectPromptTauDecayProduct_                       .push_back(ip->statusFlags().isDirectPromptTauDecayProduct());
	  isDirectPromptTauDecayProductFinalState_             .push_back(ip->isDirectPromptTauDecayProductFinalState());
	  isHardProcess_                                       .push_back(ip->isHardProcess());
	  fromHardProcessFinalState_                           .push_back(ip->fromHardProcessFinalState());
	  fromHardProcessDecayed_                              .push_back(ip->fromHardProcessDecayed());
	  
	  if(ip->mother()->pdgId() == 111){//If the photon is coming from a neutral pion
	    isPionMother_                                      .push_back(1);
	  }
	  else{
	    isPionMother_                                      .push_back(0);
	  }
	}
      }
    }
    else{
      isPhotonMatching_                            .push_back(0);
      //Setting rificulous values below, for the Photons which do not match with gen.
      phoMatchingGenPt_                            .push_back(-999);
      isPromptFinalState_                          .push_back(-999);
      isDirectHardProcessTauDecayProductFinalState_.push_back(-999);
      isDirectHadronDecayProduct_                  .push_back(-999);

      isTauDecayProduct_                           .push_back(-999);
      isPromptTauDecayProduct_                     .push_back(-999);
      isDirectTauDecayProduct_                     .push_back(-999);
      isDirectPromptTauDecayProduct_               .push_back(-999);
      isDirectPromptTauDecayProductFinalState_     .push_back(-999);
      isHardProcess_                               .push_back(-999);
      fromHardProcessFinalState_                   .push_back(-999);
      fromHardProcessDecayed_                      .push_back(-999);
      isPionMother_                                .push_back(-999);
    }
    

    nPho_++;
  }
  
}
