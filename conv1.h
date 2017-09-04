float conv1_w[20][5][5] = {
    {{-0.15741470456123352, -0.1334986835718155, 0.0035349393729120493,
      -0.014545656740665436, 0.14474698901176453},
     {0.19662915170192719, 0.10458406060934067, 0.30646541714668274,
      0.1258467733860016, 0.12757322192192078},
     {0.05110117793083191, 0.14014460146427155, -0.02791249006986618,
      0.35050880908966064, 0.31004464626312256},
     {-0.03174035996198654, 0.24436397850513458, -0.29253360629081726,
      -0.077751524746418, -0.341537207365036},
     {0.07181400805711746, -0.27683448791503906, 0.12898601591587067,
      -0.34791800379753113, -0.11557768285274506}},
    {{-0.07087861746549606, 0.06780733168125153, -0.23679903149604797,
      -0.1676802635192871, 0.03120177797973156},
     {-0.15724416077136993, 0.3406098186969757, 0.05511569604277611,
      -0.036780308932065964, 0.17171286046504974},
     {-0.2757475674152374, -0.019705655053257942, 0.22056692838668823,
      -0.008767325431108475, 0.06865222752094269},
     {0.02761717513203621, -0.1988791823387146, -0.21566812694072723,
      0.10258383303880692, -0.021545546129345894},
     {-0.25382572412490845, -0.12786895036697388, 0.2531079053878784,
      0.028154339641332626, 0.22301161289215088}},
    {{0.16566871106624603, -0.2347114086151123, 0.16663897037506104,
      0.18306277692317963, 0.12433301657438278},
     {0.0922650545835495, 0.07583015412092209, 0.2813693583011627,
      0.40574342012405396, -0.011348621919751167},
     {-0.14027509093284607, -0.19180093705654144, 0.293485164642334,
      0.25445565581321716, 0.31874534487724304},
     {0.07713115215301514, 0.04305121675133705, 0.16983380913734436,
      -0.12650223076343536, -0.250178724527359},
     {-0.07026701420545578, 0.12715895473957062, -0.010078945197165012,
      -0.21545012295246124, -0.2232772409915924}},
    {{0.23902250826358795, 0.19218400120735168, 0.12635259330272675,
      0.1645997166633606, -0.07789288461208344},
     {0.2606044113636017, 0.35590851306915283, -0.2503868639469147,
      -0.35119301080703735, -0.2005835771560669},
     {0.21290723979473114, 0.4137009084224701, 0.32514744997024536,
      0.04658959060907364, 0.11421439051628113},
     {0.10334271937608719, -0.18397819995880127, 0.024557963013648987,
      0.08155572414398193, 0.21618087589740753},
     {0.14719904959201813, 0.30341026186943054, -0.24383655190467834,
      -0.15784186124801636, -0.31493431329727173}},
    {{-0.07578609138727188, -0.22465179860591888, 0.03731502965092659,
      0.09804938733577728, 0.4002223610877991},
     {-0.0661478042602539, 0.06906906515359879, 0.16364827752113342,
      -0.016517238691449165, 0.28191474080085754},
     {-0.15774130821228027, -0.04345468804240227, -0.1957574188709259,
      -0.10568228363990784, 0.04088515788316727},
     {0.2548016309738159, -0.18225733935832977, -0.34908005595207214,
      0.29029637575149536, -0.03350608050823212},
     {-0.08417379856109619, -0.01772823929786682, -0.27350735664367676,
      -0.08035656809806824, -0.18968278169631958}},
    {{0.09407827258110046, -0.28123345971107483, 0.40553852915763855,
      -0.1424010545015335, 0.190080463886261},
     {0.26899757981300354, -0.06972971558570862, 0.1279626190662384,
      -0.04655307158827782, 0.00848036166280508},
     {0.025152990594506264, -0.07167091965675354, 0.10067498683929443,
      -0.21228691935539246, -0.12515103816986084},
     {0.19652481377124786, -0.24612867832183838, -0.30321887135505676,
      0.27176350355148315, -0.237039253115654},
     {-0.34169745445251465, -0.04344307258725166, -0.3310072124004364,
      0.09569645673036575, -0.2845461666584015}},
    {{-0.021809164434671402, -0.36404818296432495, 0.13796930015087128,
      -0.060737185180187225, -0.13644587993621826},
     {0.06006351858377457, -0.32908326387405396, 0.3079109191894531,
      0.0022908439859747887, -0.05303988605737686},
     {0.045639052987098694, -0.12081478536128998, -0.10430159419775009,
      0.3305603265762329, -0.21362334489822388},
     {-0.19001035392284393, 0.22163531184196472, 0.05606625974178314,
      0.3097924292087555, 0.2870548963546753},
     {-0.2942279577255249, 0.26879626512527466, -0.03059418499469757,
      -0.1547817587852478, 0.009918241761624813}},
    {{-0.17903557419776917, -0.10258826613426208, -0.3114202618598938,
      -0.29536548256874084, -0.3438929319381714},
     {-0.3898957371711731, -0.11147841811180115, -0.3926747143268585,
      -0.15286460518836975, 0.0722477063536644},
     {-0.27481505274772644, -0.3068278729915619, -0.3756376802921295,
      0.21514256298542023, -0.3180943429470062},
     {0.21528193354606628, -0.3870117962360382, 0.15976496040821075,
      -0.2336691915988922, 0.07975247502326965},
     {-0.1322702169418335, 0.024364648386836052, -0.28234755992889404,
      0.24145081639289856, 0.05328262224793434}},
    {{0.31002241373062134, -0.009678835980594158, -0.16958874464035034,
      -0.18910318613052368, 0.03216521441936493},
     {-0.2879636585712433, 0.1563580185174942, -0.3541584014892578,
      0.25355368852615356, 0.19136875867843628},
     {-0.2630428075790405, -0.3025679588317871, -0.28020939230918884,
      0.287967324256897, 0.3372044861316681},
     {-0.0544467493891716, -0.2862144708633423, 0.2813605070114136,
      -0.21837015450000763, -0.16648027300834656},
     {0.19072136282920837, 0.018526092171669006, 0.3395790755748749,
      0.30663809180259705, -0.054375968873500824}},
    {{0.3592645227909088, 0.29836902022361755, 0.08695780485868454,
      0.0774114653468132, -0.3049336075782776},
     {0.1890488862991333, 0.38022223114967346, 0.3036547005176544,
      -0.07623378932476044, -0.2314109355211258},
     {-0.26138797402381897, 0.30109134316444397, -0.20059610903263092,
      -0.04816913977265358, -0.3127002716064453},
     {-0.11250371485948563, 0.1821519434452057, 0.1322271078824997,
      -0.36502605676651, 0.10681869834661484},
     {0.0656425952911377, 0.2987555265426636, -0.029471388086676598,
      -0.26120784878730774, -0.2237342894077301}},
    {{-0.29023459553718567, -0.12187720090150833, 0.09316082298755646,
      -0.3235262632369995, 0.33430662751197815},
     {0.02820814587175846, 0.10320015251636505, 0.16362042725086212,
      0.1319868117570877, -0.10560112446546555},
     {0.1565045714378357, -0.2525901198387146, -0.19789163768291473,
      0.2024642527103424, -0.29320457577705383},
     {0.32252803444862366, -0.3314353823661804, 0.22879326343536377,
      -0.1774686872959137, 0.15372523665428162},
     {-0.34252727031707764, 0.10792895406484604, 0.17545638978481293,
      0.2132805436849594, -0.2475494146347046}},
    {{0.1765080839395523, -0.31216302514076233, 0.2617291510105133,
      -0.3251188397407532, 0.13012491166591644},
     {0.2572748064994812, -0.27260610461235046, 0.1782730519771576,
      0.2602241039276123, -0.270708292722702},
     {0.28166458010673523, -0.20051579177379608, -0.13841494917869568,
      -0.043565329164266586, -0.08722356706857681},
     {-0.002315970603376627, -0.2716309428215027, -0.07518935948610306,
      -0.1753445416688919, 0.18096382915973663},
     {-0.016332343220710754, 0.043154362589120865, 0.17916998267173767,
      -0.042645033448934555, 0.35344254970550537}},
    {{0.14633411169052124, 0.10256019979715347, -0.1342221051454544,
      0.3192524015903473, 0.12424565851688385},
     {-0.12157665193080902, 0.3693854808807373, 0.3802078366279602,
      -0.16694216430187225, -0.21989957988262177},
     {-0.07355587929487228, -0.28655698895454407, -0.31027984619140625,
      0.23693925142288208, 0.25868579745292664},
     {0.20393162965774536, 0.11713259667158127, 0.1532151997089386,
      0.19407017529010773, -0.10873613506555557},
     {-0.27542081475257874, -0.25567296147346497, 0.23034228384494781,
      -0.20681601762771606, 0.05116258189082146}},
    {{-0.2751367390155792, -0.2320680469274521, -0.017196862027049065,
      0.20411743223667145, 0.3176773488521576},
     {0.3083260953426361, -0.3734593093395233, -0.08729847520589828,
      -0.31607767939567566, -0.042258840054273605},
     {-0.21233472228050232, -0.22254665195941925, -0.22506819665431976,
      0.10061924904584885, -0.1840045154094696},
     {0.1457473337650299, 0.048492394387722015, -0.30033230781555176,
      0.06552407145500183, 0.1508806049823761},
     {0.09865367412567139, 0.1059264987707138, -0.31574663519859314,
      0.1398584097623825, -0.023450536653399467}},
    {{-0.18714188039302826, 0.2548518478870392, -0.06765114516019821,
      -0.06820280849933624, 0.13623295724391937},
     {-0.07027869671583176, 0.3618601858615875, 0.0932527557015419,
      0.3079819977283478, 0.06994316726922989},
     {0.02187388576567173, 0.12018341571092606, 0.3699599504470825,
      0.03585759177803993, -0.10149442404508591},
     {-0.12017308920621872, 0.2821769714355469, 0.09114377200603485,
      -0.29833799600601196, -0.06830993294715881},
     {0.13041165471076965, 0.20905981957912445, 0.06250155717134476,
      0.35654547810554504, -0.11965625733137131}},
    {{0.2032788246870041, -0.029748864471912384, 0.2138686180114746,
      0.027005445212125778, 0.4788311719894409},
     {0.024852212518453598, 0.2570211887359619, 0.37914636731147766,
      -0.12623374164104462, 0.32781457901000977},
     {0.27315235137939453, -0.17798222601413727, 0.10992277413606644,
      -0.27099961042404175, -0.3239664137363434},
     {0.08453792333602905, -0.18531760573387146, -0.40468263626098633,
      -0.38746771216392517, -0.21233804523944855},
     {-0.3251134753227234, -0.3953251838684082, -0.4260682165622711,
      -0.12434568256139755, -0.3363440930843353}},
    {{0.1417427808046341, -0.11857999861240387, -0.21653714776039124,
      -0.3067604601383209, -0.32681649923324585},
     {-0.2840665280818939, -0.16439852118492126, 0.06901231408119202,
      0.05349252372980118, -0.18045976758003235},
     {-0.3740703761577606, -0.25457534193992615, -0.06579094380140305,
      -0.27946892380714417, -0.0773455873131752},
     {-0.2930922210216522, -0.019877683371305466, -0.33515650033950806,
      0.06541475653648376, -0.33128660917282104},
     {-0.30225616693496704, -0.0928240716457367, -0.09965574741363525,
      -0.3657400906085968, 0.08559472858905792}},
    {{0.27504926919937134, -0.27945539355278015, -0.3184570372104645,
      -0.08048983663320541, -0.02222703956067562},
     {-0.2790398895740509, -0.17142042517662048, 0.003233791794627905,
      -0.2847282588481903, -0.2242741584777832},
     {0.09206964820623398, 0.025104841217398643, -0.33797547221183777,
      -0.03485977649688721, -0.16290315985679626},
     {-0.3401152193546295, 0.21018971502780914, -0.1457161158323288,
      -0.07057138532400131, 0.06805755198001862},
     {0.32529187202453613, -0.28497982025146484, 0.17802096903324127,
      -0.2761373519897461, 0.13850100338459015}},
    {{0.15928712487220764, 0.06855543702840805, 0.05480813607573509,
      0.3719468414783478, -0.00932231917977333},
     {0.17598329484462738, -0.1794312596321106, -0.31894493103027344,
      -0.21173369884490967, 0.07086646556854248},
     {-0.11196468025445938, 0.22759325802326202, -0.10085601359605789,
      0.21079540252685547, -0.1999516785144806},
     {-0.1142764687538147, 0.20215332508087158, 0.20526638627052307,
      -0.12340401858091354, -0.066200852394104},
     {-0.23226672410964966, -0.2951965630054474, -0.29569244384765625,
      0.10604061186313629, -0.23198863863945007}},
    {{-0.1859547197818756, 0.28613173961639404, 0.228526309132576,
      -0.2048298865556717, -0.1953684538602829},
     {-0.0656561627984047, 0.32131054997444153, 0.06746286153793335,
      -0.050467874854803085, 0.27332445979118347},
     {0.311701238155365, 0.0038988564629107714, 0.3844529688358307,
      0.017813578248023987, 0.3467937111854553},
     {-0.06610216200351715, 0.12058458477258682, -0.23570549488067627,
      -0.2546449303627014, -0.01996697299182415},
     {0.029059279710054398, 0.1591644436120987, 0.23266005516052246,
      0.21717040240764618, 0.19440168142318726}}};

float conv1_b[20] = {
    -0.171239107847, 0.024835107848,  -0.323164194822,  -0.189015373588,
    0.0809514373541, 0.0349024049938, -0.0147376954556, 0.19095505774,
    -0.126757115126, -0.166972339153, -0.113703228533,  -0.0388250946999,
    -0.10281611979,  0.0655098855495, -0.236635908484,  -0.0913386270404,
    0.298011749983,  0.0777169167995, -0.0506891831756, -0.175301060081};
