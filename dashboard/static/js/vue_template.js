function fullPathDecorator(f) {
  return function() {
    return '../../../experiments/' + f.call(this, arguments);
  }
}

var app = new Vue({
  el: '#app',
  data: {
    OVERVIEW: 'Overview',
    current: experiments_dir[0],
    items: experiments_dir
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
    descriptionPath: fullPathDecorator(function(){
      return this.current.dirname + '/plots/description.html';
    }),
    resultMetricsPath: function(){
      return 'https://plot.ly/~jackp/10002.embed?link=false';
    },
    schemePath: function(){
      return 'https://plot.ly/~jackp/10002.embed?link=false';
    }
  }
});
