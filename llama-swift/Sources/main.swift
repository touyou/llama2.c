import Foundation

public enum Llama {
  struct Config {
    /// transformer dimension
    let dimension: Int
    /// for ffn layers
    let hiddenDimension: Int
    /// number of layers
    let layerCount: Int
    /// number of query heads
    let headCount: Int
    /// number of key/value heads
    let keyValueHeadCount: Int
    /// vocabulary size
    let vocabularySize: Int
    /// max ksequence length
    let sequenceLength: Int

    public func toString() -> String {
      return """
        dimension: \(dimension)
        hiddenDimension: \(hiddenDimension)
        layerCount: \(layerCount)
        headCount: \(headCount)
        keyValueHeadCount: \(keyValueHeadCount)
        vocabularySize: \(vocabularySize)
        sequenceLength: \(sequenceLength)
        """
    }
  }

  struct TransformerWeights {
    private let modelData: Data
    let config: Config

    /// token embedding table
    let tokenEmbeddingTable: FloatBufferView
    /// rmsnorm weights(attention)
    let attentionRMSNormWeights: [FloatBufferView]
    /// rmsnorm weights(ffn)
    let feedForwardRMSNormWeights: [FloatBufferView]
    /// weights for matmuls(query)
    let attentionQueryWeights: [FloatBufferView]
    /// weights for matmuls(key)
    let attentionKeyWeights: [FloatBufferView]
    /// weights for matmuls(value)
    let attentionValueWeights: [FloatBufferView]
    /// weights for matmuls(output)
    let attentionOutputWeights: [FloatBufferView]
    /// weights for ffn(1st layer)
    let feedForwardLayer1Weights: [FloatBufferView]
    /// weights for ffn(2nd layer)
    let feedForwardLayer2Weights: [FloatBufferView]
    /// weights for ffn(3rd layer)
    let feedForwardLayer3Weights: [FloatBufferView]
    /// final rmsnorm weights
    let finalRMSNormWeights: FloatBufferView
    /// classifier weights(output logits)
    let outputClassifierWeights: FloatBufferView

    init(config: Config, modelData: Data, sharedWeights: Bool) throws {
      self.config = config
      self.modelData = modelData
      var currentOffset = 0
      let floatStride = MemoryLayout<Float>.stride

      let headSize = config.dimension / config.headCount

      func createView(count: Int) -> FloatBufferView {
        let byteSize = count * floatStride
        let range = currentOffset..<(currentOffset + byteSize)
        guard range.upperBound <= modelData.count else {
          fatalError("Invalid range: \(range) exceeds modelData size")
        }
        let view = FloatBufferView(data: modelData, range: range)
        currentOffset += byteSize
        return view
      }

      func createViews(layers: Int, countPerLayer: Int) -> [FloatBufferView] {
        var views = [FloatBufferView]()
        views.reserveCapacity(layers)
        for _ in 0..<layers {
          views.append(
            createView(count: countPerLayer)
          )
        }
        return views
      }

      // (vocabularySize, dimension)
      self.tokenEmbeddingTable = createView(count: config.vocabularySize * config.dimension)
      // (layerCount, dimension) rmsnorm weights
      self.attentionRMSNormWeights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension
      )
      // (layerCount, dimension, headCount * headSize)
      self.attentionQueryWeights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension * config.dimension
      )
      // (layerCount, dimension, keyValueHeadCount * headSize)
      self.attentionKeyWeights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension * (config.keyValueHeadCount * headSize)
      )
      // (layerCount, dimension, keyValueHeadCount * headSize)
      self.attentionValueWeights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension * (config.keyValueHeadCount * headSize)
      )
      // (layerCount, headCount * headSize, dimension)
      self.attentionOutputWeights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension * config.dimension
      )
      // (layerCount, dimension)
      self.feedForwardRMSNormWeights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension
      )
      // (layerCount, hiddenDimension, dimension)
      self.feedForwardLayer1Weights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension * config.hiddenDimension
      )
      // (layerCount, dimension, hiddenDimension)
      self.feedForwardLayer2Weights = createViews(
        layers: config.layerCount,
        countPerLayer: config.hiddenDimension * config.dimension
      )
      // (layerCount, hiddenDimension, dimension)
      self.feedForwardLayer3Weights = createViews(
        layers: config.layerCount,
        countPerLayer: config.dimension * config.hiddenDimension
      )
      // (dimension, )
      self.finalRMSNormWeights = createView(count: config.dimension)

      // NOTE: Skip freq_cis_real and freq_cis_imag
      let freqCisSize = config.sequenceLength * headSize / 2
      currentOffset += freqCisSize
      currentOffset += freqCisSize

      // check range
      guard currentOffset <= modelData.count else {
        fatalError("Invalid range: \(currentOffset) exceeds modelData size")
      }

      // (vocabularySize, dimension)
      if sharedWeights {
        let range = 0..<(config.vocabularySize * config.dimension * floatStride)
        self.outputClassifierWeights = FloatBufferView(data: modelData, range: range)
      } else {
        self.outputClassifierWeights = createView(count: config.vocabularySize * config.dimension)
      }

      if currentOffset != modelData.count {
        print("Warning: \(currentOffset) does not match modelData size \(modelData.count)")
      }
      print("Successfully loaded weights. Final offset: \(currentOffset)")
    }
  }
}
