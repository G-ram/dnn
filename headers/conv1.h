float conv1_w[20][5][5] = {
    {{-0.3339218497276306, -0.21203523874282837, -0.2574039101600647,
      0.30858275294303894, 0.24559977650642395},
     {0.08145096153020859, -0.09819887578487396, -0.24040769040584564,
      0.2811225354671478, 0.33004894852638245},
     {-0.014643996022641659, -0.021749606356024742, 0.2986927628517151,
      -0.13043619692325592, 0.31789630651474},
     {0.3256039321422577, 0.3452661633491516, 0.39848607778549194,
      -0.15474435687065125, 0.06624483317136765},
     {-0.0700155720114708, 0.3000979721546173, -0.0025175651535391808,
      0.24360427260398865, 0.1369488388299942}},
    {{-0.028500622138381004, -0.32400479912757874, -0.12468891590833664,
      -0.30159130692481995, -0.11251053214073181},
     {-0.26241534948349, -0.3292139172554016, 0.30219632387161255,
      -0.06943332403898239, 0.12353738397359848},
     {0.35102182626724243, 0.10390860587358475, 0.06070099025964737,
      0.1924286186695099, 0.3565579652786255},
     {-0.29659503698349, 0.3130761384963989, -0.2262268215417862,
      0.3523651659488678, 0.2773370146751404},
     {-0.11152864247560501, -0.21108537912368774, 0.314805269241333,
      -0.2644147574901581, -0.08916588872671127}},
    {{-0.08863575756549835, -0.18636806309223175, -0.2716887593269348,
      -0.37350329756736755, -0.3379266858100891},
     {-0.016331899911165237, -0.19870644807815552, -0.20751862227916718,
      0.05277136340737343, 0.3162469267845154},
     {0.18098405003547668, 0.36435818672180176, 0.13147951662540436,
      0.253508985042572, 0.08923912048339844},
     {0.21597187221050262, 0.06425357609987259, 0.23961052298545837,
      0.27708670496940613, -0.27051830291748047},
     {-0.21203681826591492, 0.15740111470222473, 0.02747846022248268,
      -0.08521442115306854, -0.2864348888397217}},
    {{-0.06010289117693901, 0.30030718445777893, -0.22642375528812408,
      -0.13456293940544128, 0.3030185103416443},
     {0.2390955686569214, -0.23013746738433838, -0.08318039774894714,
      -0.25134581327438354, 0.2860889136791229},
     {0.2451310008764267, -0.16225846111774445, 0.26628270745277405,
      -0.10111638903617859, -0.21535545587539673},
     {0.35914817452430725, 0.17431342601776123, -0.2264178991317749,
      -0.15454381704330444, -0.36777469515800476},
     {-0.13557562232017517, 0.30038997530937195, 0.31288576126098633,
      0.32373955845832825, 0.09700413048267365}},
    {{-0.23009176552295685, 0.33735090494155884, 0.0995420292019844,
      -0.11021330207586288, -0.048487402498722076},
     {0.35214245319366455, 0.23946887254714966, -0.23580780625343323,
      0.24444817006587982, -0.13353540003299713},
     {-0.24711039662361145, 0.33646783232688904, 0.0366419218480587,
      0.24770472943782806, -0.2971520721912384},
     {-0.2731907069683075, -0.2480979710817337, 0.10260801762342453,
      0.28462788462638855, -0.050306279212236404},
     {-0.004707113839685917, 0.13283400237560272, -0.26186731457710266,
      0.19706778228282928, -0.2461162656545639}},
    {{-0.3154076039791107, 0.04746264964342117, 0.16971342265605927,
      0.2869144678115845, -0.05245823413133621},
     {-0.09067437052726746, -0.13529613614082336, -0.20881648361682892,
      -0.03598029538989067, 0.3343523144721985},
     {-0.12262970209121704, -0.2709815800189972, 0.2446194887161255,
      -0.22506262362003326, 0.3580591082572937},
     {-0.05678712949156761, 0.20291493833065033, -0.004741908051073551,
      0.3709147274494171, 0.35837700963020325},
     {0.15891654789447784, -0.2896086275577545, 0.06766558438539505,
      0.04500402510166168, -0.30376195907592773}},
    {{0.3276202380657196, -0.347063809633255, -0.3389040231704712,
      -0.0639515072107315, -0.3156580924987793},
     {0.15976658463478088, 0.009405652992427349, -0.22136670351028442,
      -0.17775893211364746, 0.10445813089609146},
     {-0.04300568997859955, 0.1293436884880066, 0.19715793430805206,
      -0.29745933413505554, -0.3276076018810272},
     {0.03154967725276947, -0.31028079986572266, -0.33332377672195435,
      -0.30583876371383667, -0.32545068860054016},
     {0.017108935862779617, -0.15796935558319092, -0.0073073673993349075,
      0.16883322596549988, -0.13328275084495544}},
    {{0.2655971944332123, 0.15206226706504822, 0.39288759231567383,
      0.325754314661026, 0.14268368482589722},
     {-0.24746084213256836, 0.27396827936172485, -0.17709778249263763,
      -0.3531179130077362, -0.06828463822603226},
     {-0.3571714162826538, -0.2791127562522888, -0.2647278606891632,
      0.15116682648658752, -0.36991146206855774},
     {-0.142041876912117, -0.30925390124320984, -0.1322607398033142,
      0.14802426099777222, -0.30360034108161926},
     {-0.09039318561553955, 0.07781965285539627, 0.11939196288585663,
      -0.09199082106351852, -0.045458994805812836}},
    {{-0.010078275576233864, -0.1779727041721344, 0.3294249475002289,
      0.2778014838695526, -0.03835931420326233},
     {-0.10363749414682388, 0.21181829273700714, 0.2587948739528656,
      -0.3222521245479584, 0.03332197666168213},
     {-0.33972281217575073, -0.31190070509910583, 0.2587835192680359,
      0.030097810551524162, 0.0831141397356987},
     {0.12594464421272278, 0.15969318151474, -0.1079205572605133,
      -0.1685381978750229, -0.15662872791290283},
     {0.35418644547462463, 0.08533304184675217, -0.25666728615760803,
      0.09472876042127609, 0.07344312965869904}},
    {{-0.0408133938908577, 0.21374332904815674, 0.37743330001831055,
      0.019188595935702324, 0.08926191926002502},
     {-0.17318697273731232, 0.3556097149848938, 0.21078801155090332,
      -0.30395805835723877, -0.10916855186223984},
     {-0.136904776096344, -0.13811998069286346, 0.18926756083965302,
      0.22318647801876068, -0.031964011490345},
     {-0.10860912501811981, -0.00016364114708267152, -0.04051181674003601,
      -0.35635584592819214, -0.12351470440626144},
     {-0.08284249156713486, 0.0777188390493393, 0.2209094613790512,
      -0.37922540307044983, -0.20360395312309265}},
    {{0.3480873703956604, -0.24409952759742737, 0.23395009338855743,
      -0.1797940731048584, 0.24817630648612976},
     {-0.004027083516120911, 0.32885926961898804, 0.09606841951608658,
      -0.009959317743778229, -0.15056106448173523},
     {0.3937148451805115, -0.013008205220103264, 0.22386525571346283,
      -0.23976430296897888, 0.3053717613220215},
     {0.3467760384082794, 0.15274201333522797, 0.41019368171691895,
      0.11462955921888351, 0.25190746784210205},
     {-0.05820668488740921, 0.3003831207752228, 0.11379454284906387,
      -0.0617796964943409, 0.10205653309822083}},
    {{-0.2376049906015396, 0.3611702620983124, -0.07859116792678833,
      -0.1366218477487564, 0.12194889038801193},
     {-0.2466103881597519, 0.3107353746891022, 0.18278399109840393,
      0.3924604654312134, 0.04726242274045944},
     {-0.12368106096982956, 0.13961835205554962, -0.18216156959533691,
      0.2522318959236145, 0.011995133012533188},
     {-0.12635375559329987, 0.28138262033462524, -0.29010361433029175,
      0.2235717922449112, 0.3200782835483551},
     {-0.21124586462974548, -0.09625856578350067, -0.234911248087883,
      0.14607296884059906, -0.3392179310321808}},
    {{0.2364298403263092, -0.17492400109767914, 0.05511309951543808,
      0.20324355363845825, 0.03287740796804428},
     {0.3138427138328552, -0.29028749465942383, 0.15031825006008148,
      0.3177584707736969, 0.030629755929112434},
     {-0.14315928518772125, -0.37300240993499756, -0.23099227249622345,
      -0.20674195885658264, -0.19850414991378784},
     {-0.3698904514312744, -0.2762129604816437, -0.26673761010169983,
      0.18632923066616058, 0.0466042123734951},
     {-0.06493376940488815, 0.02632780745625496, 0.2234509140253067,
      0.29429057240486145, 0.2788161635398865}},
    {{0.2981230318546295, -0.285844624042511, -0.26199567317962646,
      0.07089807838201523, -0.271848201751709},
     {0.22279298305511475, 0.34836554527282715, -0.06939484924077988,
      0.1471845507621765, 0.2027006298303604},
     {-0.301435649394989, -0.03348406404256821, -0.29409995675086975,
      0.1102197989821434, 0.21778926253318787},
     {0.046151310205459595, -0.19324864447116852, -0.21579210460186005,
      0.2662922739982605, -0.3293502628803253},
     {-0.17045630514621735, -0.16993075609207153, -0.08234648406505585,
      0.18131142854690552, -0.26513373851776123}},
    {{-0.20897448062896729, 0.2854423224925995, -0.33943140506744385,
      0.13813291490077972, 0.167682483792305},
     {0.1999380886554718, -0.05516588315367699, 0.06204620748758316,
      -0.0692090392112732, 0.15869133174419403},
     {-0.0990966185927391, -0.18808497488498688, 0.24155685305595398,
      -0.004716123454272747, -0.16111372411251068},
     {0.007106969133019447, -0.05159170925617218, 0.1586751490831375,
      0.17690323293209076, 0.2862764894962311},
     {-0.3000427484512329, 0.1471201777458191, 0.28964924812316895,
      -0.04052522033452988, -0.11553175002336502}},
    {{0.0257350392639637, -0.15376923978328705, 0.006614576559513807,
      -0.028379565104842186, 0.37124085426330566},
     {-0.24306321144104004, 0.07441048324108124, 0.18535107374191284,
      0.1508290320634842, 0.26884788274765015},
     {0.20182253420352936, 0.3035646080970764, 0.053060971200466156,
      0.2779848873615265, 0.2437761276960373},
     {-0.3153474032878876, 0.10588983446359634, -0.1042388379573822,
      0.32076218724250793, 0.1188880056142807},
     {-0.33167240023612976, 0.07999692112207413, 0.01711316406726837,
      0.27742937207221985, 0.30791208148002625}},
    {{0.30638349056243896, 0.20524315536022186, -0.18875399231910706,
      0.058604780584573746, 0.30323028564453125},
     {0.2814265191555023, -0.0559481680393219, -0.23482900857925415,
      0.19301238656044006, -0.1335017830133438},
     {0.09632214158773422, 0.29457804560661316, 0.16062815487384796,
      0.046467628329992294, -0.07897750288248062},
     {0.3523808717727661, 0.14795039594173431, -0.021384749561548233,
      -0.05075719207525253, -0.13236068189144135},
     {-0.2216976135969162, -0.31094008684158325, -0.18409226834774017,
      -0.2898109555244446, -0.03286566585302353}},
    {{0.28535938262939453, -0.22709283232688904, -0.253701388835907,
      0.016433198004961014, -0.22311507165431976},
     {0.03677709400653839, -0.23809821903705597, -0.2306719273328781,
      -0.3834865689277649, 0.15980608761310577},
     {0.26732394099235535, -0.3581525385379791, 0.15824861824512482,
      -0.32077184319496155, -0.07978846877813339},
     {0.13663995265960693, -0.04809281975030899, 0.08989733457565308,
      -0.28188955783843994, -0.18433329463005066},
     {0.35415688157081604, 0.03945387899875641, 0.2514600157737732,
      0.060261718928813934, -0.21136337518692017}},
    {{0.26990625262260437, -0.28015100955963135, -0.19591780006885529,
      0.08417513966560364, -0.30097609758377075},
     {-0.10114877671003342, -0.22695299983024597, -0.016696497797966003,
      0.12292950600385666, -0.2672591209411621},
     {-0.3315344750881195, 0.28159773349761963, 0.16025684773921967,
      0.1693865805864334, -0.2279864400625229},
     {0.29392480850219727, 0.27323177456855774, -0.0950925350189209,
      0.30198127031326294, -0.11330602318048477},
     {0.1715499758720398, 0.05925380811095238, -0.0996052548289299,
      0.394404798746109, 0.1809781938791275}},
    {{0.15153002738952637, 0.01922064647078514, 0.35653916001319885,
      0.23534464836120605, 0.3570854365825653},
     {0.16606484353542328, 0.38699591159820557, 0.20955196022987366,
      0.1663532704114914, 0.12461697310209274},
     {-0.3701060116291046, -0.32642146944999695, -0.27478355169296265,
      -0.21394449472427368, 0.15146026015281677},
     {-0.28725379705429077, 0.07825475931167603, 0.0860631912946701,
      -0.280444473028183, -0.048525746911764145},
     {-0.012174981646239758, 0.11112663894891739, 0.24843302369117737,
      -0.12524838745594025, 0.19727596640586853}}};

float conv1_b[20] = {
    -0.2072635144,     -0.123423278332,  -0.178765207529,   -0.177219450474,
    -0.0737771317363,  -0.045762039721,  -0.00497831543908, -0.053414400667,
    -0.00976169947535, -0.0694613829255, -0.365094900131,   -0.233443886042,
    -0.188423514366,   -0.0133346589282, -0.0541695915163,  -0.18380305171,
    -0.144888520241,   -0.16285431385,   -0.163340955973,   -0.182532563806};