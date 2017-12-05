var app = new Vue({
  el: '#app',
  data: {
    LIST_NAV: 'List',
    current: '1',
    items: ['1', '2', '3']
  },
  methods: {
    setCurrentExperiment: function (item) {
      this.current = item;
    }
  },
  computed: {
    lossPlotPath: function(){
      return 'https://plot.ly/~jackp/10002.embed?link=false';
    },
    accuracyPlotPath: function(){
      return 'https://plot.ly/~jackp/10002.embed?link=false';
    },
    descriptionPath: function(){
      return 'https://plot.ly/~jackp/10002.embed?link=false';
    },
    confidencePlotPath: function(){
      return 'https://plot.ly/~jackp/10002.embed?link=false';
    },
    schemePath: function(){
      return 'https://plot.ly/~jackp/10002.embed?link=false';
    }
  }
});
