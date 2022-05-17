(this["webpackJsonpcarbon-tutorial"]=this["webpackJsonpcarbon-tutorial"]||[]).push([[0],{212:function(e,t,a){},213:function(e,t,a){},230:function(e,t){},232:function(e,t){},253:function(e,t,a){},258:function(e,t,a){"use strict";a.r(t);a(178),a(191),a(194),a(198),a(201);var n=a(0),r=a.n(n),s=a(79),l=a.n(s),i=(a(212),a(213),a(25)),c=a(26),o=a(23),m=a(31),u=a(32),h=a(137),p=a(118),d=a(121),b=a(122),v=a(139),E=a(123),f=a(270),k=a(124),g=a(125),y=a(33),x=function(e){return r.a.createElement(p.a,{render:function(t){var a=t.isSideNavExpanded,n=t.onClickSideNavExpand;return r.a.createElement(d.a,{"aria-label":"Orion Dashboard"},r.a.createElement(b.a,null),r.a.createElement(v.a,{"aria-label":"Open menu",onClick:n,isActive:a}),r.a.createElement(E.a,{element:y.b,to:"/",prefix:"Or\xedon",replace:!0},"Dashboard"),r.a.createElement(f.a,{"aria-label":"Or\xedon Dashboard"},r.a.createElement(k.a,{"aria-label":"experiments"===e.dashboard?"experiments (selected)":"experiments",menuLinkName:"Experiments"},r.a.createElement(g.a,{title:"Go to experiments visualizations",element:y.b,to:"/visualizations",replace:!0},"Visualizations"),r.a.createElement(g.a,{title:"Go to experiments status",element:y.b,to:"/status",replace:!0},"Status"),r.a.createElement(g.a,{title:"Go to experiments database",element:y.b,to:"/database",replace:!0},"Database"),r.a.createElement(g.a,{title:"Go to experiments configuration",element:y.b,to:"/configuration",replace:!0},"Configuration")),r.a.createElement(k.a,{"aria-label":"benchmarks"===e.dashboard?"benchmarks (selected)":"benchmarks",menuLinkName:"Benchmarks"},r.a.createElement(g.a,{title:"Go to benchmarks visualizations",element:y.b,to:"/benchmarks/visualizations",replace:!0},"Visualizations"),r.a.createElement(g.a,{title:"Go to benchmarks status",element:y.b,to:"/benchmarks/status",replace:!0},"Status"),r.a.createElement(g.a,{title:"Go to benchmarks database",element:y.b,to:"/benchmarks/database",replace:!0},"Database"),r.a.createElement(g.a,{title:"Go to benchmarks configuration",element:y.b,to:"/benchmarks/configuration",replace:!0},"Configuration"))))}})},O=a(66),S=a(274),w=a(72),j=a(141),N=a.n(j);function C(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{},a=arguments.length>2?arguments[2]:void 0,n=arguments.length>3?arguments[3]:void 0,r=e,s=Object.keys(t);if(s.length){r+="?";var l,i=Object(w.a)(s);try{for(i.s();!(l=i.n()).done;){var c=l.value;r+="".concat(c,"=").concat(encodeURI(t[c]))}}catch(o){i.e(o)}finally{i.f()}}N.a.get(r,(function(e){var t="";e.on("data",(function(e){t+=e})),e.on("end",(function(){a(JSON.parse(t))}))})).on("error",(function(e){n(e)}))}var _=function(){function e(t){Object(i.a)(this,e),this.baseURL=t}return Object(c.a)(e,[{key:"query",value:function(e){var t=this,a=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{};return new Promise((function(n,r){C("".concat(t.baseURL,"/").concat(e),a,n,r)}))}}]),e}(),D=window.__ORION_BACKEND__||"http://127.0.0.1:8000",A=r.a.createContext({address:D,experiment:null}),z=a(138),B=a(272),R=a(151),T=a(259).a.prefix,L=function(e){Object(m.a)(a,e);var t=Object(u.a)(a);function a(e){var n;return Object(i.a)(this,a),(n=t.call(this,e))._isMounted=!1,n.state={experiments:null,search:""},n.onSearch=n.onSearch.bind(Object(o.a)(n)),n.onUnselect=n.onUnselect.bind(Object(o.a)(n)),n}return Object(c.a)(a,[{key:"render",value:function(){return r.a.createElement(z.a,{className:"experiment-navbar",isFixedNav:!0,expanded:!0,isChildOfHeader:!1,"aria-label":"Side navigation"},r.a.createElement("div",{className:"experiments-wrapper"},r.a.createElement(B.f,{className:"experiments-list",selection:!0},r.a.createElement(B.c,null,r.a.createElement(B.e,{head:!0},r.a.createElement(B.b,{className:"experiment-cell",head:!0},"Experiment"),r.a.createElement(B.b,{head:!0},"Status"))),r.a.createElement(B.a,null,this.renderExperimentsList()))),r.a.createElement(R.a,{placeholder:"Search experiment",labelText:"Search experiment",onChange:this.onSearch}))}},{key:"renderExperimentsList",value:function(){var e,t=this;if(null===this.state.experiments)return this.renderMessageRow("Loading experiments ...");if(!this.state.experiments.length)return this.renderMessageRow("No experiment available");if(this.state.search.length){if(!(e=this.state.experiments.filter((function(e){return e.toLowerCase().indexOf(t.state.search)>=0}))).length)return this.renderMessageRow("No matching experiment")}else e=this.state.experiments;return e.map((function(e){return r.a.createElement(B.e,{label:!0,key:"row-".concat(e)},r.a.createElement(B.d,{id:"select-experiment-".concat(e),value:"row-".concat(e),title:"row-".concat(e),name:"select-experiment",onChange:function(){return t.props.onSelectExperiment(e)}}),r.a.createElement(B.b,{className:"experiment-cell"},r.a.createElement("span",{title:"unselect experiment '".concat(e,"'"),style:{visibility:t.context.experiment===e?"visible":"hidden"},onClick:function(a){return t.onUnselect(a,e,"select-experiment-".concat(e))}},r.a.createElement(S.a,{className:"".concat(T,"--structured-list-svg"),"aria-label":"unselect experiment"},r.a.createElement("title",null,"unselect experiment")))," ",r.a.createElement("span",{title:e},e)),r.a.createElement(B.b,null,r.a.createElement(O.a,null,r.a.createElement(O.a,{variant:"success",now:35,key:1}),r.a.createElement(O.a,{variant:"warning",now:20,key:2}),r.a.createElement(O.a,{variant:"danger",now:10,key:3}),r.a.createElement(O.a,{variant:"info",now:15,key:4}))))}))}},{key:"renderMessageRow",value:function(e){return r.a.createElement(B.e,null,r.a.createElement(B.b,null,e),r.a.createElement(B.b,null))}},{key:"componentDidMount",value:function(){var e=this;this._isMounted=!0,new _(this.context.address).query("experiments").then((function(t){var a=t.map((function(e){return e.name}));a.sort(),e._isMounted&&e.setState({experiments:a})})).catch((function(t){e._isMounted&&e.setState({experiments:[]})}))}},{key:"componentWillUnmount",value:function(){this._isMounted=!1}},{key:"onSearch",value:function(e){this.setState({search:(e.target.value||"").toLowerCase()})}},{key:"onUnselect",value:function(e,t,a){e.preventDefault(),document.getElementById(a).checked=!1,this.props.onSelectExperiment(null)}}]),a}(r.a.Component);L.contextType=A;var P=L,M=function(){return r.a.createElement(r.a.Fragment,null,r.a.createElement("h4",null,"Landing Page"))},U=function(){return r.a.createElement("div",null,"Status page")},I=a(152),F=a(47),H=a.n(F),W={responsive:!0};function G(e){return r.a.createElement(H.a,{id:"regret-plot",data:e.data,layout:e.layout,config:W,useResizeHandler:!0,style:{width:"100%"}})}var q={responsive:!0},J=function(e){return r.a.createElement(H.a,{id:"lpi-plot",data:e.data,layout:e.layout,config:q,useResizeHandler:!0,style:{width:"100%"}})},K={responsive:!0};function V(e){return r.a.createElement(H.a,{id:"parallel-coordinates-plot",data:e.data,layout:e.layout,config:K,useResizeHandler:!0,style:{width:"100%"}})}var $=function(e){Object(m.a)(a,e);var t=Object(u.a)(a);function a(e){var n;return Object(i.a)(this,a),(n=t.call(this,e)).state={experiment:null,regret:!1,parallel_coordinates:!1,lpi:!1,keyCount:0},n}return Object(c.a)(a,[{key:"render",value:function(){return r.a.createElement("div",{className:"bx--grid bx--grid--full-width",key:this.state.keyCount},r.a.createElement("div",{className:"bx--row"},r.a.createElement("div",{className:"bx--col-sm-16 bx--col-md-8 bx--col-lg-8 bx--col-xlg-8"},r.a.createElement("div",{className:"bx--tile plot-tile"},this.renderRegret())),r.a.createElement("div",{className:"bx--col-sm-16 bx--col-md-8 bx--col-lg-8 bx--col-xlg-8"},r.a.createElement("div",{className:"bx--tile plot-tile"},this.renderParallelCoordinates()))),r.a.createElement("div",{className:"bx--row"},r.a.createElement("div",{className:"bx--col-sm-16 bx--col-md-8 bx--col-lg-8 bx--col-xlg-8"},r.a.createElement("div",{className:"bx--tile plot-tile"},this.renderLPI()))))}},{key:"renderRegret",value:function(){return null===this.state.regret?"Loading regret plot for: ".concat(this.state.experiment," ..."):!1===this.state.regret?"Nothing to display":r.a.createElement(G,{data:this.state.regret.data,layout:this.state.regret.layout})}},{key:"renderParallelCoordinates",value:function(){return null===this.state.parallel_coordinates?"Loading parallel coordinates plot for: ".concat(this.state.experiment," ..."):!1===this.state.parallel_coordinates?"Nothing to display":r.a.createElement(V,{data:this.state.parallel_coordinates.data,layout:this.state.parallel_coordinates.layout})}},{key:"renderLPI",value:function(){return null===this.state.lpi?"Loading LPI plot for: ".concat(this.state.experiment," ..."):!1===this.state.lpi?"Nothing to display":r.a.createElement(J,{data:this.state.lpi.data,layout:this.state.lpi.layout})}},{key:"componentDidMount",value:function(){var e=this.context.experiment;null!==e&&this.loadBackendData(e)}},{key:"componentDidUpdate",value:function(e,t,a){var n=this.context.experiment;this.state.experiment!==n&&(null===n?this.setState({experiment:n,regret:!1,parallel_coordinates:!1,lpi:!1}):this.loadBackendData(n))}},{key:"loadBackendData",value:function(e){var t=this;this.setState({experiment:e,regret:null,parallel_coordinates:null,lpi:null},(function(){var a=new _(t.context.address),n=a.query("plots/regret/".concat(e)),r=a.query("plots/parallel_coordinates/".concat(e)),s=a.query("plots/lpi/".concat(e));Promise.allSettled([n,r,s]).then((function(a){var n=Object(I.a)(a,3),r=n[0],s=n[1],l=n[2],i="fulfilled"===r.status&&r.value,c="fulfilled"===s.status&&s.value,o="fulfilled"===l.status&&l.value,m=t.state.keyCount+1;t.setState({experiment:e,regret:i,parallel_coordinates:c,lpi:o,keyCount:m})}))}))}}]),a}(r.a.Component);$.contextType=A;var Q=$,X=a(65),Y=a(268),Z=a(100),ee=a(98),te=a(103),ae=a(104),ne=a(105),re=a(106),se=a(99),le=a(101),ie=a(55),ce=a(102),oe=function(e){var t=e.rows,a=e.headers;return r.a.createElement(Y.a,{rows:t,headers:a,render:function(e){var t=e.rows,a=e.headers,n=e.getHeaderProps,s=e.getRowProps,l=e.getTableProps;return r.a.createElement(Z.a,{title:"Experiment Trials",description:"Trials of selected experiments."},r.a.createElement(ee.a,l(),r.a.createElement(te.a,null,r.a.createElement(ae.a,null,r.a.createElement(ne.a,null),a.map((function(e){return r.a.createElement(re.a,n({header:e}),e.header)})))),r.a.createElement(se.a,null,t.map((function(e){return r.a.createElement(r.a.Fragment,{key:e.id},r.a.createElement(le.a,s({row:e}),e.cells.map((function(e){return r.a.createElement(ie.a,{key:e.id},e.value)}))),r.a.createElement(ce.a,{colSpan:a.length+1},r.a.createElement("p",null,"TODO: Trial detailed configuration")))})))))}})},me=[{key:"id",header:"ID"},{key:"experiment",header:"Experiment"},{key:"status",header:"Status"},{key:"created_on",header:"Created"},{key:"params",header:"Parameters"},{key:"objective",header:"Objective"}],ue=[{id:"1",experiment:"1",status:"Completed",created_on:"2020-12-01 05:05:05",updatedAt:"Date",params:[],results:[{type:"objective",name:"loss",value:1}]},{id:"2",name:"Repo 2",createdAt:"Date",updatedAt:"Date",issueCount:"123",stars:"456",links:"Links"},{id:"3",name:"Repo 3",createdAt:"Date",updatedAt:"Date",issueCount:"123",stars:"456",links:"Links"}],he=function(){return r.a.createElement("div",{className:"bx--grid bx--grid--full-width bx--grid--no-gutter database-page"},r.a.createElement("div",{className:"bx--row database-page__r1"},r.a.createElement("div",{className:"bx--col-lg-16"},r.a.createElement(oe,{headers:me,rows:(e=ue,e.map((function(e){return Object(X.a)(Object(X.a)({},e),{},{id:e.id,experiment:e.experiment,status:e.status,created_on:new Date(e.created_on).toLocaleDateString(),params:"dunno",objective:"bad"})})))}))));var e},pe=function(){return r.a.createElement("div",null,"Configuration page")},de=a(56),be=function(e){Object(m.a)(a,e);var t=Object(u.a)(a);function a(e){var n;return Object(i.a)(this,a),(n=t.call(this,e)).state={experiment:null},n.onSelectExperiment=n.onSelectExperiment.bind(Object(o.a)(n)),n}return Object(c.a)(a,[{key:"render",value:function(){return r.a.createElement(r.a.Fragment,null,r.a.createElement(A.Provider,{value:{address:D,experiment:this.state.experiment}},r.a.createElement(x,{dashboard:"experiments"}),r.a.createElement(P,{onSelectExperiment:this.onSelectExperiment}),r.a.createElement(h.a,null,this.renderPage())))}},{key:"renderPage",value:function(){switch(this.props.match.params.page||"landing"){case"landing":return r.a.createElement(M,null);case"status":return r.a.createElement(U,null);case"visualizations":return r.a.createElement(Q,null);case"database":return r.a.createElement(he,null);case"configuration":return r.a.createElement(pe,null)}}},{key:"onSelectExperiment",value:function(e){this.setState({experiment:e})}}]),a}(n.Component),ve=Object(de.e)(be),Ee=(a(253),a(269)),fe=a(265);function ke(e){if("string"===typeof e)return e;var t=Object.keys(e);if(1===t.length)return t[0];throw new Error("Cannot get algorithm name from object: ".concat(JSON.stringify(e)))}var ge=function(e){Object(m.a)(a,e);var t=Object(u.a)(a);function a(e){var n;return Object(i.a)(this,a),(n=t.call(this,e)).onChangeComboBox=n.onChangeComboBox.bind(Object(o.a)(n)),n.onSelectAlgo=n.onSelectAlgo.bind(Object(o.a)(n)),n.onSelectTask=n.onSelectTask.bind(Object(o.a)(n)),n.onSelectAssessment=n.onSelectAssessment.bind(Object(o.a)(n)),n}return Object(c.a)(a,[{key:"render",value:function(){return null===this.props.benchmarks?"":r.a.createElement(z.a,{className:"benchmark-navbar",isFixedNav:!0,expanded:!0,isChildOfHeader:!1,"aria-label":"Side navigation"},r.a.createElement(Ee.a,{onChange:this.onChangeComboBox,id:"combobox-benchmark",items:this.props.benchmarks,itemToString:function(e){return null===e?null:e.name},placeholder:"Search a benchmark ..."}),null===this.props.benchmark?"":r.a.createElement(B.f,null,this.renderAssessments(),this.renderTasks(),this.renderAlgorithms()))}},{key:"renderAssessments",value:function(){var e=this,t=this.props.benchmark,a=Object.keys(t.assessments);return a.sort(),r.a.createElement(r.a.Fragment,null,r.a.createElement(B.e,null,r.a.createElement(B.b,null,r.a.createElement("strong",null,"Assessments"))),a.map((function(t,a){return r.a.createElement(B.e,{key:a},r.a.createElement(B.b,null,r.a.createElement(fe.a,{labelText:t,id:"assessment-".concat(a),checked:e.props.assessments.has(t),onChange:function(a,n,r){return e.onSelectAssessment(t,a)}})))})))}},{key:"renderTasks",value:function(){var e=this,t=this.props.benchmark,a=Object.keys(t.tasks);return a.sort(),r.a.createElement(r.a.Fragment,null,r.a.createElement(B.e,null,r.a.createElement(B.b,null,r.a.createElement("strong",null,"Tasks"))),a.map((function(t,a){return r.a.createElement(B.e,{key:a},r.a.createElement(B.b,null,r.a.createElement(fe.a,{labelText:t,id:"task-".concat(a),checked:e.props.tasks.has(t),onChange:function(a,n,r){return e.onSelectTask(t,a)}})))})))}},{key:"renderAlgorithms",value:function(){var e=this,t=this.props.benchmark.algorithms.map((function(e){return ke(e)}));return t.sort(),r.a.createElement(r.a.Fragment,null,r.a.createElement(B.e,null,r.a.createElement(B.b,null,r.a.createElement("strong",null,"Algorithms"))),t.map((function(t,a){return r.a.createElement(B.e,{key:a},r.a.createElement(B.b,null,r.a.createElement(fe.a,{labelText:t,id:"algorithm-".concat(a),checked:e.props.algorithms.has(t),onChange:function(a,n,r){return e.onSelectAlgo(t,a)}})))})))}},{key:"onChangeComboBox",value:function(e){var t=e.selectedItem;if(null===t)this.props.onSelectBenchmark(t,new Set,new Set,new Set);else{var a=t.algorithms.map((function(e){return ke(e)}));this.props.onSelectBenchmark(t,new Set(a),new Set(Object.keys(t.tasks)),new Set(Object.keys(t.assessments)))}}},{key:"onSelectAlgo",value:function(e,t){var a=new Set(this.props.algorithms);t?a.add(e):a.delete(e),this.props.onSelectBenchmark(this.props.benchmark,a,this.props.tasks,this.props.assessments)}},{key:"onSelectTask",value:function(e,t){var a=new Set(this.props.tasks);t?a.add(e):a.delete(e),this.props.onSelectBenchmark(this.props.benchmark,this.props.algorithms,a,this.props.assessments)}},{key:"onSelectAssessment",value:function(e,t){var a=new Set(this.props.assessments);t?a.add(e):a.delete(e),this.props.onSelectBenchmark(this.props.benchmark,this.props.algorithms,this.props.tasks,a)}}]),a}(r.a.Component),ye=a(144),xe=a(109),Oe=a.n(xe),Se=function(){function e(t,a,n){Object(i.a)(this,e),this.name=t,this.data=a,this.layout=n}return Object(c.a)(e,[{key:"get",value:function(e){var t,a=[],n=Object(w.a)(this.data);try{for(n.s();!(t=n.n()).done;){var r,s=t.value,l=Object(w.a)(e);try{for(l.s();!(r=l.n()).done;){var i=r.value;0!==s.name.indexOf(i)&&0!==s.name.indexOf("".concat(i,"_"))||a.push(s)}}catch(c){l.e(c)}finally{l.f()}}}catch(c){n.e(c)}finally{n.f()}return{data:a,layout:this.layout,name:this.name}}}]),e}(),we=function(){function e(t){Object(i.a)(this,e),this.backend=new _(t),this.plots={}}return Object(c.a)(e,[{key:"get",value:function(){var e=Object(ye.a)(Oe.a.mark((function e(t,a,n,r){var s,l,i,c,o,m,u,h,p;return Oe.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if((s=((this.plots[t]||{})[a]||{})[n]||[]).length){e.next=14;break}return l="benchmarks/".concat(t,"?assessment=").concat(a,"&task=").concat(n),console.log("Loading: ".concat(this.backend.baseURL,"/").concat(l)),e.next=6,this.backend.query(l);case 6:for(i=e.sent,c=i.analysis[a][n],(o=Object.keys(c)).sort(),m=0,u=o;m<u.length;m++)h=u[m],p=JSON.parse(c[h]),s.push(new Se(h,p.data,p.layout));void 0===this.plots[t]&&(this.plots[t]={}),void 0===this.plots[t][a]&&(this.plots[t][a]={}),void 0===this.plots[t][a][n]&&(this.plots[t][a][n]=s);case 14:return e.abrupt("return",s.map((function(e){return e.get(r)})));case 15:case"end":return e.stop()}}),e,this)})));return function(t,a,n,r){return e.apply(this,arguments)}}()}]),e}(),je=new we(D),Ne=function(e){Object(m.a)(a,e);var t=Object(u.a)(a);function a(e){var n;return Object(i.a)(this,a),(n=t.call(this,e))._isMounted=!1,n.state={plots:null},n}return Object(c.a)(a,[{key:"render",value:function(){var e=this;return null===this.state.plots?"Loading plots for assessment: ".concat(this.props.assessment,", task: ").concat(this.props.task,", algorithms: ").concat(this.props.algorithms.join(", ")):!1===this.state.plots?r.a.createElement("strong",null,"Unable to load plots for assessment: ".concat(this.props.assessment,", task: ").concat(this.props.task,", algorithms: ").concat(this.props.algorithms.join(", "))):r.a.createElement("div",{className:"orion-plots"},this.state.plots.map((function(t,a){var n="plot-".concat(e.props.benchmark,"-").concat(e.props.assessment,"-").concat(e.props.task,"-").concat(t.name,"-").concat(e.props.algorithms.join("-"));return r.a.createElement(H.a,{className:"orion-plot",key:n,divId:n,data:t.data,layout:t.layout,config:{responsive:!0},useResizeHandler:!0})})))}},{key:"componentDidMount",value:function(){var e=this;this._isMounted=!0,je.get(this.props.benchmark,this.props.assessment,this.props.task,this.props.algorithms).then((function(t){e._isMounted&&e.setState({plots:t})})).catch((function(t){console.error(t),e._isMounted&&e.setState({plots:!1})}))}},{key:"componentWillUnmount",value:function(){this._isMounted=!1}},{key:"componentDidUpdate",value:function(e,t,a){if("function"===typeof Event)window.dispatchEvent(new Event("resize"));else{var n=window.document.createEvent("UIEvents");n.initUIEvent("resize",!0,!1,window,0),window.dispatchEvent(n)}}}]),a}(r.a.Component),Ce=a(273),_e=a(266),De=a(267),Ae=a(271),ze=function(e){Object(m.a)(a,e);var t=Object(u.a)(a);function a(e){var n;return Object(i.a)(this,a),(n=t.call(this,e)).onResize=n.onResize.bind(Object(o.a)(n)),n}return Object(c.a)(a,[{key:"render",value:function(){var e=this;if(null===this.props.benchmark)return r.a.createElement("div",null,r.a.createElement("h4",{className:"title-visualizations"},"No benchmark selected"));if(!this.props.assessments.size)return r.a.createElement("div",null,r.a.createElement("h4",{className:"title-visualizations"},"No assessment selected"));if(!this.props.tasks.size)return r.a.createElement("div",null,r.a.createElement("h4",{className:"title-visualizations"},"No task selected"));if(!this.props.algorithms.size)return r.a.createElement("div",null,r.a.createElement("h4",{className:"title-visualizations"},"No algorithm selected"));var t=Array.from(this.props.assessments),a=Array.from(this.props.tasks),n=Array.from(this.props.algorithms);t.sort(),a.sort(),n.sort();var s="viz-".concat(this.props.benchmark.name,"-").concat(t.join("-"),"-").concat(a.join("-"),"-").concat(n.join("-"));return r.a.createElement("div",null,r.a.createElement("h4",{className:"title-visualizations"},"Assessments"),r.a.createElement("div",{className:"assessments",id:"assessments"},t.map((function(t,l){return r.a.createElement(Ce.a,{fullWidth:!0,className:"assessment",key:"assessment-".concat(t)},r.a.createElement(_e.a,null,r.a.createElement(De.a,null,r.a.createElement(Ae.a,{className:"plot-tile"},r.a.createElement("strong",null,r.a.createElement("em",null,t))))),a.map((function(a,l){return r.a.createElement(_e.a,{key:"task-".concat(a)},r.a.createElement(De.a,{key:"task-".concat(a,"-assessment-").concat(t),className:"orion-column"},r.a.createElement(Ae.a,{className:"plot-tile"},r.a.createElement(Ne,{key:"".concat(s,"-plots-").concat(e.props.benchmark.name,"-").concat(t,"-").concat(a,"-").concat(n.join("-")),benchmark:e.props.benchmark.name,assessment:t,task:a,algorithms:n}))))})))}))))}},{key:"componentDidMount",value:function(){this.onResize(),window.addEventListener("resize",this.onResize)}},{key:"componentDidUpdate",value:function(e,t,a){this.onResize()}},{key:"componentWillUnmount",value:function(){window.removeEventListener("resize",this.onResize)}},{key:"onResize",value:function(){var e=document.getElementById("assessments");if(e)for(var t=e.offsetWidth/this.props.assessments.size,a=e.getElementsByClassName("assessment"),n=0;n<a.length;++n){for(var r=a[n],s=1,l=r.getElementsByClassName("orion-column"),i=0;i<l.length;++i){var c=l[i].getElementsByClassName("orion-plot");s<c.length&&(s=c.length)}var o=t*s;r.style.width="".concat(o,"px")}}}]),a}(r.a.Component),Be=function(){return r.a.createElement("div",null,"Benchmarks status page")},Re=function(e){var t=e.rows,a=e.headers;return r.a.createElement(Y.a,{rows:t,headers:a,render:function(e){var t=e.rows,a=e.headers,n=e.getHeaderProps,s=e.getRowProps,l=e.getTableProps;return r.a.createElement(Z.a,{title:"Experiment Trials",description:"Trials of selected experiments."},r.a.createElement(ee.a,l(),r.a.createElement(te.a,null,r.a.createElement(ae.a,null,r.a.createElement(ne.a,null),a.map((function(e){return r.a.createElement(re.a,n({header:e}),e.header)})))),r.a.createElement(se.a,null,t.map((function(e){return r.a.createElement(r.a.Fragment,{key:e.id},r.a.createElement(le.a,s({row:e}),e.cells.map((function(e){return r.a.createElement(ie.a,{key:e.id},e.value)}))),r.a.createElement(ce.a,{colSpan:a.length+1},r.a.createElement("p",null,"TODO: Trial detailed configuration")))})))))}})},Te=[{key:"id",header:"ID"},{key:"experiment",header:"Experiment"},{key:"status",header:"Status"},{key:"created_on",header:"Created"},{key:"params",header:"Parameters"},{key:"objective",header:"Objective"}],Le=[{id:"1",experiment:"1",status:"Completed",created_on:"2020-12-01 05:05:05",updatedAt:"Date",params:[],results:[{type:"objective",name:"loss",value:1}]},{id:"2",name:"Repo 2",createdAt:"Date",updatedAt:"Date",issueCount:"123",stars:"456",links:"Links"},{id:"3",name:"Repo 3",createdAt:"Date",updatedAt:"Date",issueCount:"123",stars:"456",links:"Links"}],Pe=function(){return r.a.createElement("div",{className:"bx--grid bx--grid--full-width bx--grid--no-gutter database-page"},r.a.createElement("div",{className:"bx--row database-page__r1"},r.a.createElement("div",{className:"bx--col-lg-16"},r.a.createElement(Re,{headers:Te,rows:(e=Le,e.map((function(e){return Object(X.a)(Object(X.a)({},e),{},{id:e.id,experiment:e.experiment,status:e.status,created_on:new Date(e.created_on).toLocaleDateString(),params:"dunno",objective:"bad"})})))}))));var e},Me=function(){return r.a.createElement("div",null,"Benchmarks configuration page")},Ue=function(e){Object(m.a)(a,e);var t=Object(u.a)(a);function a(e){var n;return Object(i.a)(this,a),(n=t.call(this,e))._isMounted=!1,n.state={benchmarks:null,benchmark:null,algorithms:null,tasks:null,assessments:null},n.onSelectBenchmark=n.onSelectBenchmark.bind(Object(o.a)(n)),n}return Object(c.a)(a,[{key:"render",value:function(){return r.a.createElement(r.a.Fragment,null,r.a.createElement(x,{dashboard:"benchmarks"}),null===this.state.benchmarks?r.a.createElement(h.a,null,r.a.createElement("h4",null,"Loading benchmarks ...")):0===this.state.benchmarks.length?r.a.createElement(h.a,null,r.a.createElement("h4",null,"No benchmarks available")):r.a.createElement(r.a.Fragment,null,r.a.createElement(ge,{benchmarks:this.state.benchmarks,benchmark:this.state.benchmark,algorithms:this.state.algorithms,tasks:this.state.tasks,assessments:this.state.assessments,onSelectBenchmark:this.onSelectBenchmark}),r.a.createElement(h.a,null,this.renderPage())))}},{key:"renderPage",value:function(){switch(this.props.match.params.page||"visualizations"){case"status":return r.a.createElement(Be,null);case"database":return r.a.createElement(Pe,null);case"configuration":return r.a.createElement(Me,null);case"visualizations":return r.a.createElement(ze,{benchmark:this.state.benchmark,algorithms:this.state.algorithms,tasks:this.state.tasks,assessments:this.state.assessments})}}},{key:"componentDidMount",value:function(){var e=this;this._isMounted=!0,new _(D).query("benchmarks").then((function(t){e._isMounted&&e.setState({benchmarks:t})})).catch((function(t){console.error(t),e._isMounted&&e.setState({benchmarks:[]})}))}},{key:"componentWillUnmount",value:function(){this._isMounted=!1}},{key:"onSelectBenchmark",value:function(e,t,a,n){this.setState({benchmark:e,algorithms:t,tasks:a,assessments:n})}}]),a}(n.Component),Ie=Object(de.e)(Ue),Fe=function(e){Object(m.a)(a,e);var t=Object(u.a)(a);function a(e){var n;return Object(i.a)(this,a),(n=t.call(this,e)).state={page:null},n.selectExperiments=n.selectExperiments.bind(Object(o.a)(n)),n.selectBenchmarks=n.selectBenchmarks.bind(Object(o.a)(n)),n}return Object(c.a)(a,[{key:"render",value:function(){return r.a.createElement(de.c,null,r.a.createElement(de.a,{exact:!0,path:"/",component:ve}),r.a.createElement(de.a,{exact:!0,path:"/benchmarks",component:Ie}),r.a.createElement(de.a,{exact:!0,path:"/benchmarks/:page",component:Ie}),r.a.createElement(de.a,{path:"/:page",component:ve}))}},{key:"selectExperiments",value:function(){this.setState({page:"experiments"})}},{key:"selectBenchmarks",value:function(){this.setState({page:"benchmarks"})}}]),a}(n.Component);Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));var He=a(150),We=a(149),Ge=new He.a({uri:"https://api.github.com/graphql",headers:{authorization:"Bearer ".concat(Object({NODE_ENV:"production",PUBLIC_URL:"",WDS_SOCKET_HOST:void 0,WDS_SOCKET_PATH:void 0,WDS_SOCKET_PORT:void 0,FAST_REFRESH:!0}).REACT_APP_GITHUB_PERSONAL_ACCESS_TOKEN)}});l.a.render(r.a.createElement(We.a,{client:Ge},r.a.createElement(y.a,null,r.a.createElement(Fe,null))),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()}))}},[[258,1,2]]]);
//# sourceMappingURL=main.be9daa4d.chunk.js.map